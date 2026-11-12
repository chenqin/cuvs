/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"

#include <cuvs/core/bitset.hpp>
#include <cuvs/core/bloom_filter.hpp>
#include <cuvs/core/roaring_allowlist.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

// Clustered rows, and filters that pass whole clusters: the case PASSING_TWO_HOP is for.
constexpr int64_t n_rows      = 20000;
constexpr int64_t per_cluster = 100;
constexpr int64_t n_queries   = 32;
constexpr int64_t k           = 10;
// What a result slot holds when the search has no row for it.
constexpr uint32_t invalid_row = 0xffffffffu;

class CagraFilterTwoHopTest : public ::testing::Test {
 protected:
  explicit CagraFilterTwoHopTest(int64_t n_dim = 16) : n_dim(n_dim) {}

  void SetUp() override
  {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    const int64_t n_clusters = n_rows / per_cluster;
    std::vector<float> centers(n_clusters * n_dim);
    for (auto& x : centers) {
      x = uniform(gen);
    }
    host_rows.resize(n_rows * n_dim);
    for (int64_t i = 0; i < n_rows; i++) {
      for (int64_t d = 0; d < n_dim; d++) {
        host_rows[i * n_dim + d] = centers[(i / per_cluster) * n_dim + d] + noise(gen);
      }
    }
    host_queries.resize(n_queries * n_dim);
    for (int64_t q = 0; q < n_queries; q++) {
      const int64_t c = gen() % n_clusters;
      for (int64_t d = 0; d < n_dim; d++) {
        host_queries[q * n_dim + d] = centers[c * n_dim + d] + 4 * noise(gen);
      }
    }

    auto stream = raft::resource::get_cuda_stream(res);
    dataset.emplace(raft::make_device_matrix<float, int64_t>(res, n_rows, n_dim));
    queries.emplace(raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim));
    raft::copy(dataset->data_handle(), host_rows.data(), host_rows.size(), stream);
    raft::copy(queries->data_handle(), host_queries.data(), host_queries.size(), stream);
    // Filter 0 passes the clusters whose number is 3 modulo 10 and filter 1 those whose number is
    // 7 modulo 10: a tenth of the rows each.
    for (int f = 0; f < 2; f++) {
      std::vector<uint32_t> words((n_rows + 31) / 32, 0u);
      for (int64_t i = 0; i < n_rows; i++) {
        if ((i / per_cluster) % 10 == 3 + 4 * f) { words[i / 32] |= 1u << (i % 32); }
      }
      add_filter(std::move(words));
    }

    cagra::index_params index_params;
    index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
    padded.emplace(res, raft::make_const_mdspan(dataset->view()));
    index.emplace(cagra::build(res, index_params, padded->view));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index.reset();
    padded.reset();
    bits.clear();
    queries.reset();
    dataset.reset();
    raft::resource::sync_stream(res);
  }

  // Makes the rows whose bits are set in `words` a filter, on the host and on the device, and
  // returns its number.
  int add_filter(std::vector<uint32_t> words)
  {
    bits.push_back(raft::make_device_vector<uint32_t, int64_t>(res, words.size()));
    raft::copy(
      bits.back().data_handle(), words.data(), words.size(), raft::resource::get_cuda_stream(res));
    host_bits.push_back(std::move(words));
    return static_cast<int>(host_bits.size()) - 1;
  }

  // Filter -1 passes every row.
  bool passes(int f, uint32_t row) const
  {
    return row < n_rows && (f < 0 || ((host_bits[f][row / 32] >> (row % 32)) & 1u));
  }

  std::set<uint32_t> exact(int f, int64_t q) const
  {
    std::vector<std::pair<float, uint32_t>> scored;
    for (int64_t i = 0; i < n_rows; i++) {
      if (!passes(f, static_cast<uint32_t>(i))) { continue; }
      float dist = 0;
      for (int64_t d = 0; d < n_dim; d++) {
        const float x = host_rows[i * n_dim + d] - host_queries[q * n_dim + d];
        dist += x * x;
      }
      scored.emplace_back(dist, static_cast<uint32_t>(i));
    }
    const auto n = std::min<int64_t>(k, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + n, scored.end());
    std::set<uint32_t> top;
    for (int64_t i = 0; i < n; i++) {
      top.insert(scored[i].second);
    }
    return top;
  }

  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(int f)
  {
    return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      cuvs::core::bitset_view<uint32_t, int64_t>(bits[f].data_handle(), n_rows));
  }

  // One pointer per query to the words of its filter, on the device; a null pointer for filter -1.
  template <typename FilterOf>
  raft::device_vector<uintptr_t, int64_t> query_pointers(FilterOf filter_of)
  {
    std::vector<uintptr_t> host_pointers(n_queries);
    for (int64_t q = 0; q < n_queries; q++) {
      const int f      = filter_of(q);
      host_pointers[q] = f < 0 ? 0 : reinterpret_cast<uintptr_t>(bits[f].data_handle());
    }
    auto pointers = raft::make_device_vector<uintptr_t, int64_t>(res, n_queries);
    raft::copy(pointers.data_handle(),
               host_pointers.data(),
               host_pointers.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return pointers;
  }

  std::vector<uint32_t> search(cagra::search_params const& params,
                               cuvs::neighbors::filtering::base_filter const& sample_filter)
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
    cagra::search(res,
                  params,
                  *index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  sample_filter);
    std::vector<uint32_t> out(n_queries * k);
    raft::copy(
      out.data(), neighbors.data_handle(), out.size(), raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return out;
  }

  std::vector<uint32_t> search(cagra::search_params const& params, int f)
  {
    return search(params, filter(f));
  }

  // Every row returned for query q passes filter f and is returned once; returns how many of the
  // exact answers were found.
  int64_t hits(std::vector<uint32_t> const& found, int64_t q, int f)
  {
    const auto truth = exact(f, q);
    std::set<uint32_t> seen;
    int64_t n = 0;
    for (int64_t i = 0; i < k; i++) {
      const uint32_t row = found[q * k + i];
      if (row == invalid_row) { continue; }
      EXPECT_TRUE(passes(f, row)) << "query " << q << " returned row " << row;
      EXPECT_TRUE(seen.insert(row).second) << "query " << q << " returned row " << row << " twice";
      n += truth.count(row);
    }
    return n;
  }

  // Every returned row passes its query's filter; returns the share of the exact answers found.
  template <typename FilterOf>
  double check(std::vector<uint32_t> const& found, FilterOf filter_of)
  {
    int64_t n = 0;
    for (int64_t q = 0; q < n_queries; q++) {
      n += hits(found, q, filter_of(q));
    }
    return static_cast<double>(n) / static_cast<double>(n_queries * k);
  }

  static cagra::search_params two_hop_params()
  {
    cagra::search_params params;
    params.algo                 = cagra::search_algo::MULTI_CTA;
    params.itopk_size           = 256;
    params.max_iterations       = 20;
    params.num_random_samplings = 4;
    params.filter_expansion     = cagra::filter_expansion_mode::PASSING_TWO_HOP;
    return params;
  }

  static const uint32_t* const* as_query_bitsets(raft::device_vector<uintptr_t, int64_t> const& p)
  {
    return reinterpret_cast<const uint32_t* const*>(p.data_handle());
  }

  // max_queries splits the call into batches of five queries, so a batch after the first finds
  // its pointers at an offset. The filters alternate with the query number, which gives a query
  // of every other batch another filter than the query at its place in the first batch has. One
  // query past the first batch has a null pointer and is not filtered.
  void one_bitset_per_query_in_batches()
  {
    constexpr int64_t batch = 5;
    // The unfiltered query is one whose nearest rows pass neither filter, so that filtering it by
    // a pointer of another query would not go unnoticed.
    int64_t unfiltered = -1;
    for (int64_t q = batch; q < n_queries && unfiltered < 0; q++) {
      const auto nearest = exact(-1, q);
      if (std::none_of(nearest.begin(), nearest.end(), [this](uint32_t row) {
            return passes(0, row) || passes(1, row);
          })) {
        unfiltered = q;
      }
    }
    ASSERT_GE(unfiltered, batch);
    auto filter_of = [unfiltered](int64_t q) {
      return q == unfiltered ? -1 : static_cast<int>(q % 2);
    };

    auto pointers        = query_pointers(filter_of);
    auto params          = two_hop_params();
    params.max_queries   = batch;
    params.query_bitsets = as_query_bitsets(pointers);
    for (auto mode :
         {cagra::filter_expansion_mode::PASSING_TWO_HOP, cagra::filter_expansion_mode::SCORE_ALL}) {
      params.filter_expansion = mode;
      params.filtering_rate   = 0.9f;
      params.max_iterations   = mode == cagra::filter_expansion_mode::SCORE_ALL ? 0 : 20;
      const auto found        = search(params, 0);
      const double recall     = check(found, filter_of);
      EXPECT_GE(recall, mode == cagra::filter_expansion_mode::SCORE_ALL ? 0.5 : 0.9);
      EXPECT_GE(hits(found, unfiltered, -1), k - 1) << "the query with a null pointer";
    }
  }

  // One allowlist per query, which the linked filter reads by the query's number: in batches of
  // five as above, with filters that alternate with the query number.
  void one_allowlist_per_query_in_batches()
  {
    std::vector<cuvs::core::roaring_allowlist> allowlists;
    for (int f = 0; f < 2; f++) {
      std::vector<uint32_t> ids;
      for (uint32_t row = 0; row < n_rows; row++) {
        if (passes(f, row)) { ids.push_back(row); }
      }
      allowlists.push_back(cuvs::core::roaring_allowlist::from_ids(
        res,
        n_rows,
        raft::make_host_vector_view<const uint32_t, int64_t>(ids.data(), ids.size()),
        true));
    }
    std::vector<cuvs::core::roaring_allowlist_view> views;
    for (int64_t q = 0; q < n_queries; q++) {
      views.push_back(allowlists[q % 2].view());
    }
    cuvs::neighbors::filtering::roaring_bitmap_filter roaring_filter(res, views);

    auto params        = two_hop_params();
    params.max_queries = 5;
    const auto found   = search(params, roaring_filter);
    EXPECT_GE(check(found, [](int64_t q) { return static_cast<int>(q % 2); }), 0.9);
  }

  raft::device_resources res;
  const int64_t n_dim;
  std::vector<float> host_rows, host_queries;
  std::vector<std::vector<uint32_t>> host_bits;
  std::optional<raft::device_matrix<float, int64_t>> dataset, queries;
  std::vector<raft::device_vector<uint32_t, int64_t>> bits;
  std::optional<cuvs::neighbors::test::padded_device_matrix_for_cagra<float>> padded;
  std::optional<cagra::index<float, uint32_t>> index;
};

// Rows of 17 floats are not a multiple of 16 bytes wide. The search copies such queries to padded
// rows and launches them one at a time, each launch with the number of its query as the offset.
class CagraFilterTwoHopOddWidthTest : public CagraFilterTwoHopTest {
 protected:
  CagraFilterTwoHopOddWidthTest() : CagraFilterTwoHopTest(17) {}
};

TEST_F(CagraFilterTwoHopTest, FindsFilteredNeighbors)
{
  const auto found = search(two_hop_params(), 0);
  EXPECT_GE(check(found, [](int64_t) { return 0; }), 0.9);
}

TEST_F(CagraFilterTwoHopTest, ScoreAllStillWorks)
{
  auto params             = two_hop_params();
  params.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
  params.max_iterations   = 0;
  const auto found        = search(params, 0);
  EXPECT_GE(check(found, [](int64_t) { return 0; }), 0.5);
}

// The second hop reads the graph in rounds whose shape depends on the thread block size, and the
// distances of the children it finds are computed by teams of threads.
TEST_F(CagraFilterTwoHopTest, ThreadBlockAndTeamSizes)
{
  for (size_t thread_block_size : {64, 128, 256, 512, 1024}) {
    for (size_t team_size : {0, 8, 16, 32}) {
      auto params              = two_hop_params();
      params.thread_block_size = thread_block_size;
      params.team_size         = team_size;
      const auto found         = search(params, 0);
      EXPECT_GE(check(found, [](int64_t) { return 0; }), 0.9)
        << "thread_block_size " << thread_block_size << ", team_size " << team_size;
    }
  }
}

TEST_F(CagraFilterTwoHopTest, OneBitsetPerQuery)
{
  // Even queries take filter 0, odd ones filter 1; the filter object's own bitset is not read.
  auto filter_of       = [](int64_t q) { return static_cast<int>(q % 2); };
  auto pointers        = query_pointers(filter_of);
  auto params          = two_hop_params();
  params.query_bitsets = as_query_bitsets(pointers);
  for (auto mode :
       {cagra::filter_expansion_mode::PASSING_TWO_HOP, cagra::filter_expansion_mode::SCORE_ALL}) {
    params.filter_expansion = mode;
    params.filtering_rate   = 0.9f;
    params.max_iterations   = mode == cagra::filter_expansion_mode::SCORE_ALL ? 0 : 20;
    const auto found        = search(params, 0);
    const double recall     = check(found, filter_of);
    EXPECT_GE(recall, mode == cagra::filter_expansion_mode::SCORE_ALL ? 0.5 : 0.9);
  }
}

TEST_F(CagraFilterTwoHopTest, OneBitsetPerQueryInBatches) { one_bitset_per_query_in_batches(); }

TEST_F(CagraFilterTwoHopOddWidthTest, OneBitsetPerQuery) { one_bitset_per_query_in_batches(); }

TEST_F(CagraFilterTwoHopTest, QueryBitsetsNeedMultiCta)
{
  auto pointers        = raft::make_device_vector<uintptr_t, int64_t>(res, n_queries);
  auto params          = two_hop_params();
  params.algo          = cagra::search_algo::SINGLE_CTA;
  params.query_bitsets = as_query_bitsets(pointers);
  EXPECT_THROW(search(params, 0), raft::exception);
}

// SCORE_ALL lengthens its candidate list by the filtering rate, which cannot be counted from the
// filter object's bitset when every query brings its own.
TEST_F(CagraFilterTwoHopTest, QueryBitsetsNeedFilteringRateForScoreAll)
{
  auto pointers           = query_pointers([](int64_t q) { return static_cast<int>(q % 2); });
  auto params             = two_hop_params();
  params.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
  params.query_bitsets    = as_query_bitsets(pointers);
  EXPECT_THROW(search(params, 0), raft::logic_error);
}

// The per-query bitsets stand in for the bitset of a bitset_filter of 32-bit words.
TEST_F(CagraFilterTwoHopTest, QueryBitsetsNeedBitsetFilter)
{
  auto pointers        = query_pointers([](int64_t q) { return static_cast<int>(q % 2); });
  auto params          = two_hop_params();
  params.query_bitsets = as_query_bitsets(pointers);

  // No filter object. When the algorithm is left to AUTO, whichever one it picks refuses.
  cuvs::neighbors::filtering::none_sample_filter no_filter;
  EXPECT_THROW(search(params, no_filter), raft::logic_error);
  params.algo = cagra::search_algo::AUTO;
  EXPECT_THROW(search(params, no_filter), raft::exception);
  params.algo = cagra::search_algo::MULTI_CTA;

  // A filter of another type.
  cuvs::neighbors::filtering::udf_filter udf_filter(
    "__device__ bool cuvs_filter_udf(uint32_t, source_index_t, void*) { return true; }");
  EXPECT_THROW(search(params, udf_filter), raft::logic_error);

  // A bitset whose words are not read as 32-bit words.
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> bytes_filter(
    cuvs::core::bitset_view<uint32_t, int64_t>(bits[0].data_handle(), n_rows, 8));
  EXPECT_THROW(search(params, bytes_filter), raft::logic_error);
}

// Six rows of one cluster pass, each within two hops of the others.
TEST_F(CagraFilterTwoHopTest, FewerPassingRowsThanNeighbors)
{
  const std::vector<uint32_t> rows = {4203, 4217, 4230, 4244, 4268, 4291};
  std::vector<uint32_t> words((n_rows + 31) / 32, 0u);
  for (auto row : rows) {
    words[row / 32] |= 1u << (row % 32);
  }
  const auto found = search(two_hop_params(), add_filter(std::move(words)));
  for (int64_t q = 0; q < n_queries; q++) {
    // The passing rows come first, nearest first, and the slots after them hold no row.
    std::vector<uint32_t> returned(found.begin() + q * k, found.begin() + q * k + rows.size());
    std::sort(returned.begin(), returned.end());
    EXPECT_EQ(returned, rows) << "query " << q;
    for (int64_t i = rows.size(); i < k; i++) {
      EXPECT_EQ(found[q * k + i], invalid_row) << "query " << q << ", neighbor " << i;
    }
  }
}

TEST_F(CagraFilterTwoHopTest, NoPassingRow)
{
  const int f      = add_filter(std::vector<uint32_t>((n_rows + 31) / 32, 0u));
  const auto found = search(two_hop_params(), f);
  EXPECT_EQ(std::count(found.begin(), found.end(), invalid_row), n_queries * k);
}

// A filter that is not a bitset of 32-bit words is called for every node the walk examines, and
// the start nodes are drawn by testing random nodes with it.
TEST_F(CagraFilterTwoHopTest, BloomFilter)
{
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<uint32_t> host_all_rows(n_rows);
  std::iota(host_all_rows.begin(), host_all_rows.end(), 0u);
  std::vector<uint32_t> host_added;
  for (auto row : host_all_rows) {
    if (passes(0, row)) { host_added.push_back(row); }
  }
  auto all_rows = raft::make_device_vector<uint32_t, int64_t>(res, n_rows);
  auto added    = raft::make_device_vector<uint32_t, int64_t>(res, host_added.size());
  raft::copy(all_rows.data_handle(), host_all_rows.data(), host_all_rows.size(), stream);
  raft::copy(added.data_handle(), host_added.data(), host_added.size(), stream);

  // The rows of filter 0. A false positive would be a passing row on its own among rejected
  // ones, which the walk reaches only when it starts there: the filter is sized to have none.
  cuvs::core::bloom_filter bloom(
    res, n_rows, static_cast<float>(host_added.size()) / static_cast<float>(n_rows), 1e-6f);
  bloom.add(res, raft::make_const_mdspan(added.view()));

  // What the bloom filter passes is what it says it contains.
  auto contained = raft::make_device_vector<uint8_t, int64_t>(res, n_rows);
  bloom.contains(res, raft::make_const_mdspan(all_rows.view()), contained.view());
  std::vector<uint8_t> host_contained(n_rows);
  raft::copy(host_contained.data(), contained.data_handle(), host_contained.size(), stream);
  raft::resource::sync_stream(res);
  std::vector<uint32_t> words((n_rows + 31) / 32, 0u);
  for (int64_t i = 0; i < n_rows; i++) {
    if (host_contained[i]) { words[i / 32] |= 1u << (i % 32); }
  }
  const int f = add_filter(std::move(words));

  const auto found = search(two_hop_params(), cuvs::neighbors::filtering::bloom_filter(bloom));
  EXPECT_GE(check(found, [f](int64_t) { return f; }), 0.9);
}

TEST_F(CagraFilterTwoHopTest, RoaringBitmapFilter) { one_allowlist_per_query_in_batches(); }

TEST_F(CagraFilterTwoHopOddWidthTest, RoaringBitmapFilter) { one_allowlist_per_query_in_batches(); }

// The words of filter 0 declared as a bitset of bytes pass the same rows, but the kernel does not
// read such a bitset itself: it calls the linked filter.
TEST_F(CagraFilterTwoHopTest, BitsetOfBytes)
{
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> bytes_filter(
    cuvs::core::bitset_view<uint32_t, int64_t>(bits[0].data_handle(), n_rows, 8));
  const auto found = search(two_hop_params(), bytes_filter);
  EXPECT_GE(check(found, [](int64_t) { return 0; }), 0.9);
}

// PASSING_TWO_HOP and query_bitsets take MULTI_CTA when the algorithm is left to AUTO. The batch
// is made large enough for AUTO to take SINGLE_CTA when a search asks for neither: SINGLE_CTA
// would walk as SCORE_ALL, which does not reach this recall here, and would refuse the bitsets.
TEST_F(CagraFilterTwoHopTest, AutoSelectsMultiCta)
{
  auto params        = two_hop_params();
  params.algo        = cagra::search_algo::AUTO;
  params.max_queries = 2 * raft::resource::get_device_properties(res).multiProcessorCount;
  EXPECT_GE(check(search(params, 0), [](int64_t) { return 0; }), 0.9);

  auto filter_of          = [](int64_t q) { return static_cast<int>(q % 2); };
  auto pointers           = query_pointers(filter_of);
  params.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
  params.max_iterations   = 0;
  params.filtering_rate   = 0.9f;
  params.query_bitsets    = as_query_bitsets(pointers);
  EXPECT_GE(check(search(params, 0), filter_of), 0.5);
}

// Another algorithm than MULTI_CTA walks as SCORE_ALL, with a warning.
TEST_F(CagraFilterTwoHopTest, OtherAlgorithmsScoreAll)
{
  for (auto algo : {cagra::search_algo::SINGLE_CTA, cagra::search_algo::MULTI_KERNEL}) {
    auto params           = two_hop_params();
    params.algo           = algo;
    params.max_iterations = 0;
    for (auto row : search(params, 0)) {
      // These algorithms do not all mark a slot without a row as MULTI_CTA does.
      if (row < n_rows) { EXPECT_TRUE(passes(0, row)) << "returned row " << row; }
    }
  }
}

}  // namespace
}  // namespace cuvs::neighbors::cagra
