/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../../../src/neighbors/detail/cagra/fp8_scan.hpp"
#include "../cagra_padded_build_helpers.cuh"

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/composite/index.hpp>
#include <cuvs/preprocessing/quantize/fp8.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

// More queries than one scan kernel takes (64).
constexpr int64_t n_queries    = 100;
constexpr uint32_t invalid_row = 0xffffffffu;

// Expects call to throw a raft::exception whose message contains reason.
template <typename Call>
void expect_refused(Call&& call, std::string const& reason)
{
  try {
    call();
    ADD_FAILURE() << "not refused: " << reason;
  } catch (const raft::exception& e) {
    EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
  }
}

class CagraFp8ScanTest : public ::testing::TestWithParam<int64_t> {
 protected:
  // By default rows past the 65536 the fused selection needs, with one filter passing more than
  // that and one passing less.
  explicit CagraFp8ScanTest(int64_t n_rows = 140000) : n_rows(n_rows) {}

  void SetUp() override
  {
    if (!detail::fp8_scan::is_supported(res)) {
      GTEST_SKIP() << "this build holds no FP8 scan kernels for the device";
    }
    n_dim = GetParam();
    std::mt19937 gen(4321);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    // Unit rows (cosine as L2 for the 1-past-16 dimensions: a last coordinate of 1 / sqrt(2)
    // everywhere, alone in its 16-dimension block, so scan_skip_last_block may leave it out).
    const bool constant_tail = n_dim % 16 == 1;
    auto make_rows = [&](int64_t count, std::vector<half>& out, std::vector<float>& back) {
      out.resize(count * n_dim);
      back.resize(count * n_dim);
      const int64_t free_dims = constant_tail ? n_dim - 1 : n_dim;
      for (int64_t i = 0; i < count; i++) {
        double norm = 0;
        std::vector<float> v(n_dim);
        for (int64_t d = 0; d < free_dims; d++) {
          v[d] = normal(gen);
          norm += double(v[d]) * v[d];
        }
        const double scale = constant_tail ? std::sqrt(0.5 / norm) : 1.0 / std::sqrt(norm);
        for (int64_t d = 0; d < free_dims; d++) {
          v[d] = static_cast<float>(v[d] * scale);
        }
        if (constant_tail) { v[n_dim - 1] = static_cast<float>(std::sqrt(0.5)); }
        for (int64_t d = 0; d < n_dim; d++) {
          out[i * n_dim + d]  = __float2half(v[d]);
          back[i * n_dim + d] = __half2float(out[i * n_dim + d]);
        }
      }
    };
    make_rows(n_rows, host_rows, host_rows_f);
    make_rows(n_queries, host_queries, host_queries_f);

    auto stream = raft::resource::get_cuda_stream(res);
    dataset.emplace(raft::make_device_matrix<half, int64_t>(res, n_rows, n_dim));
    queries.emplace(raft::make_device_matrix<half, int64_t>(res, n_queries, n_dim));
    raft::copy(dataset->data_handle(), host_rows.data(), host_rows.size(), stream);
    raft::copy(queries->data_handle(), host_queries.data(), host_queries.size(), stream);

    // Filter 0 passes rows whose number is not a multiple of 3 (two thirds: the fused selection),
    // filter 1 every 20th row (5%: the full selection), filter 2 five rows, filter 3 none; of 3000
    // rows, every filter takes the full selection, and filter 2 passes one row.
    add_filter([](int64_t i) { return i % 3 != 0; });
    add_filter([](int64_t i) { return i % 20 == 7; });
    add_filter([](int64_t i) { return i % 28000 == 11; });
    add_filter([](int64_t) { return false; });

    cagra::index_params index_params;
    index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
    padded.emplace(res, raft::make_const_mdspan(dataset->view()));
    index.emplace(cagra::build(res, index_params, padded->view));

    codes.emplace(res, n_rows, static_cast<uint32_t>(n_dim));
    cuvs::preprocessing::quantize::fp8::transform(
      res, raft::make_const_mdspan(dataset->view()), *codes);
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index.reset();
    padded.reset();
    codes.reset();
    bits.clear();
    queries.reset();
    dataset.reset();
    raft::resource::sync_stream(res);
  }

  // Makes the rows that pass a filter, as a bitset and as a list, and returns its number.
  template <typename Pass>
  int add_filter(Pass pass)
  {
    std::vector<uint32_t> words((n_rows + 31) / 32, 0u);
    std::vector<uint32_t> listed;
    for (int64_t i = 0; i < n_rows; i++) {
      if (pass(i)) {
        words[i / 32] |= 1u << (i % 32);
        listed.push_back(static_cast<uint32_t>(i));
      }
    }
    bits.push_back(raft::make_device_vector<uint32_t, int64_t>(res, words.size()));
    raft::copy(
      bits.back().data_handle(), words.data(), words.size(), raft::resource::get_cuda_stream(res));
    row_lists.push_back(raft::make_device_vector<uint32_t, int64_t>(res, listed.size()));
    raft::copy(row_lists.back().data_handle(),
               listed.data(),
               listed.size(),
               raft::resource::get_cuda_stream(res));
    host_bits.push_back(std::move(words));
    host_lists.push_back(std::move(listed));
    return static_cast<int>(host_bits.size()) - 1;
  }

  void attach() { index->update_scan_codes(res, codes->view()); }

  // Filter -1 passes every row.
  bool passes(int f, uint32_t row) const
  {
    return row < n_rows && (f < 0 || ((host_bits[f][row / 32] >> (row % 32)) & 1u));
  }

  double exact_distance(int64_t q, uint32_t row) const
  {
    double dist = 0;
    for (int64_t d = 0; d < n_dim; d++) {
      const double x = double(host_rows_f[row * n_dim + d]) - host_queries_f[q * n_dim + d];
      dist += x * x;
    }
    return dist;
  }

  // The k nearest passing rows of query q, by exact distance over the FP16 rows.
  std::vector<uint32_t> exact(int f, int64_t q, int64_t k) const
  {
    std::vector<std::pair<double, uint32_t>> scored;
    for (int64_t i = 0; i < n_rows; i++) {
      if (passes(f, static_cast<uint32_t>(i))) {
        scored.emplace_back(exact_distance(q, static_cast<uint32_t>(i)), static_cast<uint32_t>(i));
      }
    }
    const auto n = std::min<int64_t>(k, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + n, scored.end());
    std::vector<uint32_t> top;
    for (int64_t i = 0; i < n; i++) {
      top.push_back(scored[i].second);
    }
    return top;
  }

  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(int f)
  {
    return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      cuvs::core::bitset_view<uint32_t, int64_t>(bits[f].data_handle(), n_rows));
  }

  // The rows filter f passes, as a list.
  cuvs::neighbors::filtering::row_list_filter row_list(int f)
  {
    return cuvs::neighbors::filtering::row_list_filter(
      raft::make_const_mdspan(row_lists[f].view()));
  }

  template <typename OutputIdxT>
  struct search_result {
    std::vector<OutputIdxT> rows;
    std::vector<float> distances;
  };
  using result = search_result<uint32_t>;

  template <typename OutputIdxT = uint32_t>
  auto search(cagra::index<half, uint32_t> const& idx,
              cagra::search_params const& params,
              cuvs::neighbors::filtering::base_filter const& sample_filter,
              int64_t k) -> search_result<OutputIdxT>
  {
    auto neighbors = raft::make_device_matrix<OutputIdxT, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
    cagra::search(res,
                  params,
                  idx,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  sample_filter);
    search_result<OutputIdxT> out{std::vector<OutputIdxT>(n_queries * k),
                                  std::vector<float>(n_queries * k)};
    raft::copy(out.rows.data(),
               neighbors.data_handle(),
               out.rows.size(),
               raft::resource::get_cuda_stream(res));
    raft::copy(out.distances.data(),
               distances.data_handle(),
               out.distances.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return out;
  }

  result search(cagra::search_params const& params,
                cuvs::neighbors::filtering::base_filter const& sample_filter,
                int64_t k)
  {
    return search(*index, params, sample_filter, k);
  }

  // An index of the fixture's rows constructed from a metric and given only its dataset: it has
  // rows and no graph.
  auto make_index_without_graph() -> cagra::index<half, uint32_t>
  {
    cagra::index<half, uint32_t> out(res, cuvs::distance::DistanceType::L2Expanded);
    out = cagra::update_dataset(res, std::move(out), padded->view);
    return out;
  }

  static cagra::search_params scan_params(uint32_t candidates = 0)
  {
    cagra::search_params params;
    params.filtered_route  = filtered_route_mode::SCAN;
    params.scan_candidates = candidates;
    return params;
  }

  // Every returned row passes filter f, once, in ascending order of its exact squared distance;
  // slots past the passing rows are empty. Returns the share of the exact answers found.
  double check(result const& found, int f, int64_t k)
  {
    int64_t hits = 0, wanted = 0;
    for (int64_t q = 0; q < n_queries; q++) {
      const auto truth = exact(f, q, k);
      const std::set<uint32_t> truth_set(truth.begin(), truth.end());
      std::set<uint32_t> seen;
      float last = -1.0f;
      for (int64_t i = 0; i < k; i++) {
        const uint32_t row = found.rows[q * k + i];
        const float dist   = found.distances[q * k + i];
        if (i >= static_cast<int64_t>(truth.size())) {
          EXPECT_EQ(row, invalid_row) << "query " << q << " slot " << i;
          EXPECT_EQ(dist, FLT_MAX) << "query " << q << " slot " << i;
          continue;
        }
        if (row == invalid_row) {
          ADD_FAILURE() << "query " << q << " slot " << i << " is empty";
          continue;
        }
        EXPECT_TRUE(passes(f, row)) << "query " << q << " returned row " << row;
        EXPECT_TRUE(seen.insert(row).second)
          << "query " << q << " returned row " << row << " twice";
        const double want = exact_distance(q, row);
        EXPECT_NEAR(dist, want, 1e-4 * std::max(1.0, want)) << "query " << q << " row " << row;
        EXPECT_GE(dist, last) << "query " << q << " slot " << i;
        last = dist;
        hits += truth_set.count(row);
      }
      wanted += truth.size();
    }
    return wanted ? static_cast<double>(hits) / static_cast<double>(wanted) : 1.0;
  }

  raft::device_resources res;
  const int64_t n_rows;
  int64_t n_dim = 0;
  std::vector<half> host_rows, host_queries;
  std::vector<float> host_rows_f, host_queries_f;
  std::vector<std::vector<uint32_t>> host_bits, host_lists;
  std::optional<raft::device_matrix<half, int64_t>> dataset, queries;
  std::vector<raft::device_vector<uint32_t, int64_t>> bits, row_lists;
  std::optional<cuvs::neighbors::test::padded_device_matrix_for_cagra<half>> padded;
  std::optional<cagra::index<half, uint32_t>> index;
  std::optional<cuvs::preprocessing::quantize::fp8::quantizer> codes;
};

// With every candidate it may keep, the scan is exact over the passing rows.
TEST_P(CagraFp8ScanTest, ExactWithManyCandidates)
{
  attach();
  for (int f : {0, 1}) {
    for (int64_t k : {10, 300}) {
      EXPECT_EQ(check(search(scan_params(4096), filter(f), k), f, k), 1.0)
        << "filter " << f << " k " << k;
    }
  }
}

// The default candidates (four times k): FP8 order loses little.
TEST_P(CagraFp8ScanTest, DefaultCandidates)
{
  attach();
  for (int f : {0, 1}) {
    for (int64_t k : {10, 100, 1000}) {
      const double recall = check(search(scan_params(), filter(f), k), f, k);
      EXPECT_GE(recall, 0.95) << "filter " << f << " k " << k;
      RAFT_LOG_INFO("recall: dims %ld filter %d k %ld: %.4f",
                    static_cast<long>(n_dim),
                    f,
                    static_cast<long>(k),
                    recall);
    }
  }
}

TEST_P(CagraFp8ScanTest, SkipLastBlock)
{
  if (n_dim % 16 != 1) { GTEST_SKIP() << "no single-coordinate last block"; }
  attach();
  auto all_candidates                 = scan_params(4096);
  all_candidates.scan_skip_last_block = true;
  EXPECT_EQ(check(search(all_candidates, filter(0), 10), 0, 10), 1.0);
  auto default_candidates                 = scan_params();
  default_candidates.scan_skip_last_block = true;
  EXPECT_GE(check(search(default_candidates, filter(1), 100), 1, 100), 0.95);
}

// A row_list_filter of the rows a bitset passes is the same filter. The full selection (filter 1)
// is deterministic, so the results are the same; the fused one (filter 0) appends its candidates
// with atomics, so among candidates tied in FP8 distance at the cut, which are kept can change from
// call to call: there the recall is the same.
TEST_P(CagraFp8ScanTest, RowListEqualsBitset)
{
  attach();
  for (int f : {0, 1}) {
    const auto params    = scan_params();
    const auto by_bitset = search(params, filter(f), 50);
    const auto by_list   = search(params, row_list(f), 50);
    if (f == 1) {
      EXPECT_EQ(by_bitset.rows, by_list.rows) << "filter " << f;
      EXPECT_EQ(by_bitset.distances, by_list.distances) << "filter " << f;
    } else {
      const double a = check(by_bitset, f, 50), b = check(by_list, f, 50);
      EXPECT_GE(std::min(a, b), 0.95);
      EXPECT_NEAR(a, b, 0.01);
    }
  }
}

TEST_P(CagraFp8ScanTest, Unfiltered)
{
  attach();
  EXPECT_EQ(
    check(search(scan_params(4096), cuvs::neighbors::filtering::none_sample_filter{}, 10), -1, 10),
    1.0);
}

TEST_P(CagraFp8ScanTest, FewerPassingRowsThanNeighbors)
{
  attach();
  check(search(scan_params(), filter(2), 10), 2, 10);
  check(search(scan_params(), filter(3), 10), 3, 10);
}

TEST_P(CagraFp8ScanTest, AutoRoute)
{
  // Without codes AUTO walks the graph.
  auto params = cagra::search_params{};
  (void)search(params, filter(1), 10);
  attach();
  // 5% passing: AUTO takes the scan, as SCAN does.
  EXPECT_EQ(search(params, filter(1), 10).rows, search(scan_params(), filter(1), 10).rows);
  // GRAPH never scans.
  params.filtered_route = filtered_route_mode::GRAPH;
  (void)search(params, filter(1), 10);
}

// Rows equal to query 0, more of them than the fused selection keeps (32768): they all score the
// same FP8 distance, so the sample's cutoff lands on them and the kept count is out of range, and
// the scan falls back to the full selection. The results are still the exact ones.
TEST_P(CagraFp8ScanTest, FallbackWhenTheCutoffFails)
{
  constexpr size_t copies = 40000;
  std::vector<uint32_t> copied;
  for (uint32_t i = 0; copied.size() < copies; i++) {
    if (passes(0, i)) { copied.push_back(i); }
  }
  for (const uint32_t row : copied) {
    for (int64_t d = 0; d < n_dim; d++) {
      host_rows[row * n_dim + d]   = host_queries[d];
      host_rows_f[row * n_dim + d] = host_queries_f[d];
    }
  }
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(dataset->data_handle(), host_rows.data(), host_rows.size(), stream);
  // The index reranks from its own (padded) rows when it holds a copy.
  auto rows = index->dataset().data_const_view();
  if (rows.data_handle() != dataset->data_handle()) {
    raft::copy_matrix(const_cast<half*>(rows.data_handle()),
                      static_cast<size_t>(rows.stride(0)),
                      host_rows.data(),
                      static_cast<size_t>(n_dim),
                      static_cast<size_t>(n_dim),
                      static_cast<size_t>(n_rows),
                      stream);
  }
  cuvs::preprocessing::quantize::fp8::transform(
    res, raft::make_const_mdspan(dataset->view()), *codes);
  attach();

  constexpr int64_t k = 10;
  const auto found    = search(scan_params(), filter(0), k);
  const std::set<uint32_t> copy_set(copied.begin(), copied.end());
  for (int64_t i = 0; i < k; i++) {
    EXPECT_EQ(copy_set.count(found.rows[i]), 1u) << "slot " << i << " row " << found.rows[i];
    EXPECT_NEAR(found.distances[i], 0.0f, 1e-6f) << "slot " << i;
  }
  // Query 0's answers are any k of its copies; the other queries' are exact as before.
  EXPECT_GE(check(found, 0, k), 0.95);

  // With scan_retry the scan does not fall back by itself: it flags query 0 (and any other query
  // whose cutoff failed), and a caller keeps the others' results and searches the flagged ones
  // again with scan_full_selection. Together they are as exact as the fallback's.
  auto retry        = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto params       = scan_params();
  params.scan_retry = retry.data_handle();
  // A sentinel no flag takes: every query's flag must be written.
  RAFT_CUDA_TRY(
    cudaMemsetAsync(retry.data_handle(), 0xff, n_queries * sizeof(uint32_t), stream.get()));
  auto merged = search(params, filter(0), k);
  std::vector<uint32_t> flags(n_queries);
  raft::copy(flags.data(), retry.data_handle(), n_queries, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(flags[0], 1u);
  // Only queries near the copies can see their cutoff fail.
  EXPECT_LE(std::count(flags.begin(), flags.end(), 1u), n_queries / 10);
  params.scan_full_selection = true;
  const auto again           = search(params, filter(0), k);
  for (int64_t q = 0; q < n_queries; q++) {
    EXPECT_LE(flags[q], 1u) << "query " << q;
    if (flags[q] != 0) {
      std::copy_n(again.rows.begin() + q * k, k, merged.rows.begin() + q * k);
      std::copy_n(again.distances.begin() + q * k, k, merged.distances.begin() + q * k);
    }
  }
  for (int64_t i = 0; i < k; i++) {
    EXPECT_EQ(copy_set.count(merged.rows[i]), 1u) << "slot " << i << " row " << merged.rows[i];
    EXPECT_NEAR(merged.distances[i], 0.0f, 1e-6f) << "slot " << i;
  }
  EXPECT_GE(check(merged, 0, k), 0.95);
  raft::copy(flags.data(), retry.data_handle(), n_queries, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(std::count_if(flags.begin(), flags.end(), [](uint32_t x) { return x != 0; }), 0);
}

// On ordinary data the sampled cutoff (filter 0, with the default candidates) flags no query, and
// writes every flag; the full selection (filter 1 always, filter 0 when asked) writes 0s too and is
// exact; the graph route clears the flags.
TEST_P(CagraFp8ScanTest, FullSelectionAndRetryFlags)
{
  attach();
  auto stream = raft::resource::get_cuda_stream(res);
  auto retry  = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  std::vector<uint32_t> flags(n_queries);
  auto set_sentinel = [&] {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(retry.data_handle(), 0xff, n_queries * sizeof(uint32_t), stream.get()));
  };
  auto flagged = [&] {
    raft::copy(flags.data(), retry.data_handle(), n_queries, stream);
    raft::resource::sync_stream(res);
    return std::count_if(flags.begin(), flags.end(), [](uint32_t x) { return x != 0; });
  };
  for (int f : {0, 1}) {
    auto params       = scan_params();
    params.scan_retry = retry.data_handle();
    set_sentinel();
    EXPECT_GE(check(search(params, filter(f), 10), f, 10), 0.95) << "filter " << f;
    EXPECT_EQ(flagged(), 0) << "filter " << f;
    params.scan_full_selection = true;
    params.scan_candidates     = 4096;
    set_sentinel();
    EXPECT_EQ(check(search(params, filter(f), 10), f, 10), 1.0) << "filter " << f;
    EXPECT_EQ(flagged(), 0) << "filter " << f;
  }
  auto graph           = cagra::search_params{};
  graph.filtered_route = filtered_route_mode::GRAPH;
  graph.scan_retry     = retry.data_handle();
  set_sentinel();
  (void)search(graph, filter(1), 10);
  EXPECT_EQ(flagged(), 0);
}

// AUTO hands what the scan cannot run (here k above 4096) to the graph, which may refuse it for its
// own reasons (k above itopk_size), but not with the scan's; SCAN says why the scan cannot.
TEST_P(CagraFp8ScanTest, AutoDeclinesWhatTheScanCannotTake)
{
  attach();
  try {
    (void)search(cagra::search_params{}, filter(1), 5000);
  } catch (const raft::exception& e) {
    const std::string what = e.what();
    EXPECT_EQ(what.find("filtered scan"), std::string::npos) << what;
    EXPECT_EQ(what.find("filtered_route SCAN"), std::string::npos) << what;
  }
  EXPECT_THROW((void)search(scan_params(), filter(1), 5000), raft::exception);
}

TEST_P(CagraFp8ScanTest, ScanNeedsCodes)
{
  EXPECT_THROW(search(scan_params(), filter(1), 10), raft::exception);
  attach();
  index->reset_scan_codes();
  EXPECT_THROW(search(scan_params(), filter(1), 10), raft::exception);
}

// More than 1024 candidates are ranked by final_rank, which rounds their count up to a multiple of
// 4 for its shared memory: 1501 candidates from scan_candidates, after the fused selection (filter
// 0) and after the full one (filter 1), and 1030 from a filter that passes 1030 rows, every one of
// them a candidate, so that the results are exact.
TEST_P(CagraFp8ScanTest, FinalRankCandidateCounts)
{
  attach();
  constexpr int64_t k = 300;
  for (int f : {0, 1}) {
    const double recall = check(search(scan_params(1501), filter(f), k), f, k);
    EXPECT_GE(recall, 0.95) << "filter " << f;
    RAFT_LOG_INFO("recall: dims %ld filter %d k %ld with 1501 candidates: %.4f",
                  static_cast<long>(n_dim),
                  f,
                  static_cast<long>(k),
                  recall);
  }
  const int f = add_filter([](int64_t i) { return i % 20 == 7 && i < 20 * 1030; });
  ASSERT_EQ(host_lists[f].size(), size_t{1030});
  EXPECT_EQ(check(search(scan_params(), filter(f), k), f, k), 1.0);
  EXPECT_EQ(check(search(scan_params(), row_list(f), k), f, k), 1.0);
}

INSTANTIATE_TEST_SUITE_P(Dims,
                         CagraFp8ScanTest,
                         ::testing::Values(int64_t{33}, int64_t{65}, int64_t{128}));

// The same filters over 3000 rows, which every filter passes too few of for the fused selection:
// for the cases that need no more rows.
class CagraFp8ScanSmallTest : public CagraFp8ScanTest {
 protected:
  CagraFp8ScanSmallTest() : CagraFp8ScanTest(3000) {}
};

// A row_list_filter that walks the graph is the bitset of its rows: under GRAPH, and under AUTO
// when more than scan_max_pass_rate of the rows pass (filter 0 passes two thirds of them, filter 1
// a twentieth), the results are those of the same rows as a bitset_filter, which AUTO walks too.
// SINGLE_CTA walks each query in one CTA, so that equal walks give equal results.
TEST_P(CagraFp8ScanSmallTest, RowListOnTheGraph)
{
  attach();
  auto graph           = cagra::search_params{};
  graph.algo           = search_algo::SINGLE_CTA;
  graph.filtered_route = filtered_route_mode::GRAPH;
  for (int f : {0, 1}) {
    auto declining               = cagra::search_params{};
    declining.algo               = search_algo::SINGLE_CTA;
    declining.scan_max_pass_rate = f == 0 ? 0.25f : 0.01f;
    const auto by_bitset         = search(graph, filter(f), 10);
    for (auto const& params : {graph, declining}) {
      const auto by_list = search(params, row_list(f), 10);
      EXPECT_EQ(by_list.rows, by_bitset.rows) << "filter " << f;
      EXPECT_EQ(by_list.distances, by_bitset.distances) << "filter " << f;
    }
    EXPECT_EQ(search(declining, filter(f), 10).rows, by_bitset.rows) << "filter " << f;
  }
}

// An index of rows and no graph is searched by the scan only. With codes, AUTO scans it, filtered
// or not, as SCAN scans an index of the same rows with a graph. GRAPH fails, and so does AUTO
// without codes or where the scan cannot run (here k above 4096): there is no graph to walk.
TEST_P(CagraFp8ScanSmallTest, IndexWithoutAGraph)
{
  auto free_index = make_index_without_graph();
  ASSERT_EQ(static_cast<int64_t>(free_index.size()), n_rows);
  ASSERT_EQ(free_index.graph().extent(0), 0);
  auto graph           = cagra::search_params{};
  graph.filtered_route = filtered_route_mode::GRAPH;

  expect_refused([&] { search(free_index, cagra::search_params{}, filter(1), 10); }, "no graph");
  free_index.update_scan_codes(res, codes->view());
  expect_refused([&] { search(free_index, graph, filter(1), 10); }, "no graph");
  expect_refused([&] { search(free_index, cagra::search_params{}, filter(1), 5000); }, "no graph");

  // Every passing row a candidate: the results are exact.
  auto all_candidates            = cagra::search_params{};
  all_candidates.scan_candidates = 4096;
  const cuvs::neighbors::filtering::none_sample_filter no_filter{};
  EXPECT_EQ(check(search(free_index, all_candidates, no_filter, 10), -1, 10), 1.0);
  for (int f : {0, 1}) {
    EXPECT_EQ(check(search(free_index, all_candidates, filter(f), 10), f, 10), 1.0)
      << "filter " << f;
    EXPECT_EQ(check(search(free_index, all_candidates, row_list(f), 10), f, 10), 1.0)
      << "filter " << f;
  }

  // The full selection (filter 1) is deterministic: the same codes and rows give the same results.
  attach();
  const auto with_graph    = search(scan_params(), filter(1), 10);
  const auto without_graph = search(free_index, cagra::search_params{}, filter(1), 10);
  EXPECT_EQ(without_graph.rows, with_graph.rows);
  EXPECT_EQ(without_graph.distances, with_graph.distances);
}

// Nor can an index of rows and no graph be written, as a CAGRA index or in the hnswlib format:
// there is no graph to write, and nothing is written.
TEST_P(CagraFp8ScanSmallTest, IndexWithoutAGraphIsNotSerialized)
{
  auto free_index = make_index_without_graph();
  free_index.update_scan_codes(res, codes->view());
  std::ostringstream os;
  expect_refused([&] { cagra::serialize(res, os, free_index); }, "no graph");
  expect_refused([&] { cagra::serialize(res, os, free_index, false); }, "no graph");
  expect_refused([&] { cagra::serialize_to_hnswlib(res, os, free_index); }, "no graph");
  EXPECT_TRUE(os.str().empty());
}

// update_scan_codes takes one row of codes and one scale for each row of the index, rows at least
// the dimension rounded up to 16 bytes wide and a multiple of 16, codes 16-byte and scales 4-byte
// aligned, for an FP16 index; it refuses anything else and leaves the index without codes.
TEST_P(CagraFp8ScanSmallTest, UpdateScanCodesRefuses)
{
  namespace fp8        = cuvs::preprocessing::quantize::fp8;
  const auto full      = codes->view();
  const int64_t width  = full.codes.extent(1);
  const uint8_t* bytes = full.codes.data_handle();
  const float* scales  = full.row_scales.data_handle();
  ASSERT_EQ(width, static_cast<int64_t>(fp8::get_encoded_row_length(static_cast<uint32_t>(n_dim))));
  // Views of the codes and scales with other shapes and bases: refused before they are read.
  auto view = [](const uint8_t* b, int64_t rows, int64_t w, const float* s, int64_t n_scales) {
    return fp8::quantizer_view{raft::make_device_matrix_view<const uint8_t, int64_t>(b, rows, w),
                               raft::make_device_vector_view<const float, int64_t>(s, n_scales)};
  };
  auto refused = [&](fp8::quantizer_view const& v, std::string const& reason) {
    expect_refused([&] { index->update_scan_codes(res, v); }, reason);
    EXPECT_FALSE(index->scan_codes().has_value()) << reason;
  };
  const auto* misaligned_scales =
    reinterpret_cast<const float*>(reinterpret_cast<const char*>(scales) + 2);
  refused(view(bytes, n_rows - 1, width, scales, n_rows - 1), "for an index of");
  refused(view(bytes, n_rows, width, scales, n_rows - 1), "for an index of");
  refused(view(bytes, n_rows, width - 16, scales, n_rows), "not a multiple of 16 of at least");
  refused(view(bytes, n_rows, width + 8, scales, n_rows), "not a multiple of 16 of at least");
  refused(view(bytes + 8, n_rows, width, scales, n_rows), "16-byte aligned");
  refused(view(bytes, n_rows, width, misaligned_scales, n_rows), "4-byte aligned");

  cagra::index<float, uint32_t> fp32_index(res);
  expect_refused([&] { fp32_index.update_scan_codes(res, full); }, "FP16");

  index->update_scan_codes(res, full);
  ASSERT_TRUE(index->scan_codes().has_value());
  EXPECT_EQ(index->scan_codes()->codes.data_handle(), bytes);
  EXPECT_EQ(index->scan_codes()->row_scales.data_handle(), scales);
  index->reset_scan_codes();
  EXPECT_FALSE(index->scan_codes().has_value());
}

// Codes describe the rows they were made from: moving the index keeps them, and a new dataset
// (update_dataset, extend) drops them. Without them AUTO walks the graph (SINGLE_CTA, so that equal
// walks give equal results) and SCAN fails.
TEST_P(CagraFp8ScanSmallTest, CodesDroppedWithTheDataset)
{
  attach();
  auto automatic       = cagra::search_params{};
  automatic.algo       = search_algo::SINGLE_CTA;
  auto graph           = automatic;
  graph.filtered_route = filtered_route_mode::GRAPH;
  const auto walked    = search(graph, filter(1), 10);

  auto moved = std::move(*index);
  EXPECT_TRUE(moved.scan_codes().has_value());
  *index = std::move(moved);
  ASSERT_TRUE(index->scan_codes().has_value());

  *index = cagra::update_dataset(res, std::move(*index), padded->view);
  EXPECT_FALSE(index->scan_codes().has_value());
  expect_refused([&] { search(scan_params(), filter(1), 10); }, "no scan codes");
  EXPECT_EQ(search(automatic, filter(1), 10).rows, walked.rows);

  // The rows of the index and copies of its first 64, padded as the index's rows are.
  attach();
  constexpr int64_t extra = 64;
  auto stream             = raft::resource::get_cuda_stream(res);
  auto extended           = raft::make_device_matrix<half, int64_t>(res, n_rows + extra, n_dim);
  raft::copy(extended.data_handle(), dataset->data_handle(), n_rows * n_dim, stream);
  raft::copy(
    extended.data_handle() + n_rows * n_dim, dataset->data_handle(), extra * n_dim, stream);
  cuvs::neighbors::test::padded_device_matrix_for_cagra<half> extended_rows(
    res, raft::make_const_mdspan(extended.view()));
  cagra::extend(res, cagra::extend_params{}, extended_rows.view, n_rows, *index);
  EXPECT_EQ(static_cast<int64_t>(index->size()), n_rows + extra);
  EXPECT_FALSE(index->scan_codes().has_value());
  expect_refused(
    [&] { search(scan_params(), cuvs::neighbors::filtering::none_sample_filter{}, 10); },
    "no scan codes");
  // The index now views rows of this test.
  index.reset();
}

// An unknown filtered_route fails every search; SCAN fails, saying why, on a bitset of fewer rows
// than the index and on a row list of more.
TEST_P(CagraFp8ScanSmallTest, RefusesBadRoutesAndFilters)
{
  auto unknown           = cagra::search_params{};
  unknown.filtered_route = static_cast<filtered_route_mode>(7);
  expect_refused([&] { search(unknown, filter(1), 10); }, "unknown filtered_route");
  attach();
  expect_refused([&] { search(unknown, filter(1), 10); }, "unknown filtered_route");

  const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> short_bitset(
    cuvs::core::bitset_view<uint32_t, int64_t>(bits[1].data_handle(), n_rows - 1));
  expect_refused([&] { search(scan_params(), short_bitset, 10); }, "a bitset of");
  // Not read: the search is refused first.
  auto too_many = raft::make_device_vector<uint32_t, int64_t>(res, n_rows + 1);
  const cuvs::neighbors::filtering::row_list_filter long_list(
    raft::make_const_mdspan(too_many.view()));
  expect_refused([&] { search(scan_params(), long_list, 10); }, "a row_list_filter of");
}

// The scan writes int64 neighbors as it writes uint32 ones, and the slots past the rows that pass
// get 0xFFFFFFFF widened (4294967295) and the largest float: filter 1 passes more rows than k,
// filter 2 one row and filter 3 none.
TEST_P(CagraFp8ScanSmallTest, Int64Neighbors)
{
  attach();
  constexpr int64_t k     = 10;
  constexpr int64_t empty = 4294967295;
  for (int f : {1, 2, 3}) {
    const auto narrow = search(*index, scan_params(), filter(f), k);
    EXPECT_GE(check(narrow, f, k), 0.95) << "filter " << f;
    const auto wide = search<int64_t>(*index, scan_params(), filter(f), k);
    int64_t n_empty = 0;
    for (size_t i = 0; i < wide.rows.size(); i++) {
      const int64_t want = narrow.rows[i] == invalid_row ? empty : int64_t{narrow.rows[i]};
      EXPECT_EQ(wide.rows[i], want) << "filter " << f << " slot " << i;
      EXPECT_EQ(wide.distances[i], narrow.distances[i]) << "filter " << f << " slot " << i;
      n_empty += wide.rows[i] == empty;
    }
    const auto passing = static_cast<int64_t>(host_lists[f].size());
    EXPECT_EQ(n_empty, n_queries * (k - std::min(k, passing))) << "filter " << f;
  }
}

// A composite index of several children searches each child by walking its graph: it refuses SCAN,
// scan_retry and a row list, and walks under AUTO, also with a filter a single index would scan
// (filter 1); children without a graph therefore cannot be searched, even with codes.
TEST_P(CagraFp8ScanSmallTest, CompositeOfTwoChildren)
{
  attach();
  using composite_index = cuvs::neighbors::composite::composite_index<half, uint32_t>;
  constexpr int64_t k   = 10;
  auto neighbors        = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
  auto distances        = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
  auto search_children  = [&](composite_index const& composite,
                             cagra::search_params const& params,
                             cuvs::neighbors::filtering::base_filter const& sample_filter) {
    composite.search(res,
                     params,
                     raft::make_const_mdspan(queries->view()),
                     neighbors.view(),
                     distances.view(),
                     sample_filter);
    raft::resource::sync_stream(res);
  };
  const composite_index composite({&*index, &*index});
  expect_refused([&] { search_children(composite, scan_params(), filter(1)); }, "composite");
  auto retry            = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto with_retry       = cagra::search_params{};
  with_retry.scan_retry = retry.data_handle();
  expect_refused([&] { search_children(composite, with_retry, filter(1)); }, "composite");
  expect_refused([&] { search_children(composite, cagra::search_params{}, row_list(1)); },
                 "composite");
  EXPECT_NO_THROW(search_children(composite, cagra::search_params{}, filter(1)));

  auto free_index = make_index_without_graph();
  free_index.update_scan_codes(res, codes->view());
  const composite_index without_graphs({&free_index, &free_index});
  expect_refused([&] { search_children(without_graphs, cagra::search_params{}, filter(1)); },
                 "no graph");
}

INSTANTIATE_TEST_SUITE_P(Dims,
                         CagraFp8ScanSmallTest,
                         ::testing::Values(int64_t{33}, int64_t{65}, int64_t{128}));

}  // namespace
}  // namespace cuvs::neighbors::cagra
