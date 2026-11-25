/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"

#include "../../../src/neighbors/detail/cagra/filter_expansion_choice.cuh"

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

// The dataset is clustered and each filter passes whole clusters, so the rows that pass a filter
// are close to each other. This is the case PASSING_TWO_HOP targets.
constexpr int64_t n_rows      = 20000;
constexpr int64_t per_cluster = 100;
constexpr int64_t n_queries   = 32;
constexpr int64_t k           = 10;
// MULTI_CTA writes this index to a result slot that it could not fill.
constexpr uint32_t invalid_row = 0xffffffffu;
// Minimum recall against the exact filtered top-k rows. PASSING_TWO_HOP is held to the higher
// floor because the filters pass whole clusters, the case it targets. SCORE_ALL is held to a
// lower floor on this dataset: the tests that search with it check that the filter is applied to
// each query.
constexpr double min_recall_two_hop   = 0.9;
constexpr double min_recall_score_all = 0.5;
// Filters 0 and 1 each pass one cluster in ten, so each filters out 90% of the rows.
constexpr float known_filtering_rate = 0.9f;

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

  // Adds a filter that passes the rows whose bits are set in `words`. Keeps the words on the host
  // and on the device and returns the index of the new filter.
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
    std::vector<std::pair<float, uint32_t>> ranked;
    for (int64_t i = 0; i < n_rows; i++) {
      if (!passes(f, static_cast<uint32_t>(i))) { continue; }
      float dist = 0;
      for (int64_t d = 0; d < n_dim; d++) {
        const float x = host_rows[i * n_dim + d] - host_queries[q * n_dim + d];
        dist += x * x;
      }
      ranked.emplace_back(dist, static_cast<uint32_t>(i));
    }
    const auto n = std::min<int64_t>(k, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + n, ranked.end());
    std::set<uint32_t> top;
    for (int64_t i = 0; i < n; i++) {
      top.insert(ranked[i].second);
    }
    return top;
  }

  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(int f)
  {
    return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      cuvs::core::bitset_view<uint32_t, int64_t>(bits[f].data_handle(), n_rows));
  }

  // Returns a device array with one pointer per query to the words of its filter, or a null
  // pointer for filter -1.
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

  // Expects every row returned for query q to pass filter f and to be returned once. Returns how
  // many of the exact top-k rows were found.
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

  // Expects every returned row to pass its query's filter (see hits()). Returns the recall against
  // the exact filtered top-k rows.
  template <typename FilterOf>
  double filtered_recall(std::vector<uint32_t> const& found, FilterOf filter_of)
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

  // max_queries = 5 splits the search into batches of five queries, so each batch after the first
  // must read its pointers at the batch offset. The filters alternate between consecutive queries
  // and the batch size is odd, so in every second batch a query has a different filter than the
  // query at the same position in the first batch: reading the pointers without the offset would
  // apply the wrong filter. One query after the first batch has a null pointer and must be
  // searched unfiltered.
  void one_bitset_per_query_in_batches()
  {
    constexpr int64_t batch = 5;
    // The unfiltered query is one whose exact nearest rows pass neither filter: if the search
    // applied the bitset of another query to it, none of those rows could be returned.
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
      params.filtering_rate   = known_filtering_rate;
      params.max_iterations   = mode == cagra::filter_expansion_mode::SCORE_ALL ? 0 : 20;
      const auto found        = search(params, 0);
      const double recall     = filtered_recall(found, filter_of);
      EXPECT_GE(recall,
                mode == cagra::filter_expansion_mode::SCORE_ALL ? min_recall_score_all
                                                                : min_recall_two_hop);
      // The unfiltered query may miss at most one of its k exact nearest rows.
      EXPECT_GE(hits(found, unfiltered, -1), k - 1) << "the query with a null pointer";
    }
  }

  // A roaring_bitmap_filter with one allowlist per query, searched in batches of five as above.
  // The linked filter looks up the allowlist by the query id plus the batch offset.
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
    EXPECT_GE(filtered_recall(found, [](int64_t q) { return static_cast<int>(q % 2); }),
              min_recall_two_hop);
  }

  raft::resources res;
  const int64_t n_dim;
  std::vector<float> host_rows;
  std::vector<float> host_queries;
  std::vector<std::vector<uint32_t>> host_bits;
  std::optional<raft::device_matrix<float, int64_t>> dataset;
  std::optional<raft::device_matrix<float, int64_t>> queries;
  std::vector<raft::device_vector<uint32_t, int64_t>> bits;
  std::optional<cuvs::neighbors::test::padded_device_matrix_for_cagra<float>> padded;
  std::optional<cagra::index<float, uint32_t>> index;
};

// Rows of 17 floats are not a multiple of 16 bytes wide. The search copies such queries to padded
// rows and launches them one at a time, each launch with its query id as the batch offset.
class CagraFilterTwoHopOddWidthTest : public CagraFilterTwoHopTest {
 protected:
  CagraFilterTwoHopOddWidthTest() : CagraFilterTwoHopTest(17) {}
};

TEST_F(CagraFilterTwoHopTest, FindsFilteredNeighbors)
{
  const auto found = search(two_hop_params(), 0);
  EXPECT_GE(filtered_recall(found, [](int64_t) { return 0; }), min_recall_two_hop);
}

TEST_F(CagraFilterTwoHopTest, ScoreAllFindsFilteredNeighbors)
{
  auto params             = two_hop_params();
  params.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
  params.max_iterations   = 0;
  const auto found        = search(params, 0);
  EXPECT_GE(filtered_recall(found, [](int64_t) { return 0; }), min_recall_score_all);
}

// filter_expansion_mode::AUTO samples rows that pass the filter and counts their graph neighbors
// that pass it too. Filter 0 passes whole clusters, so a passing row's neighbors mostly pass, and
// AUTO selects PASSING_TWO_HOP.
TEST_F(CagraFilterTwoHopTest, AutoChoosesTwoHopForClusteredFilter)
{
  const auto view = filter(0).bitset_view_;
  const auto estimate =
    detail::estimate_passing_neighbors(res, index->graph(), index->source_indices(), view);
  EXPECT_GE(estimate.samples, detail::kFilterExpansionMinSamples);
  EXPECT_GE(estimate.mean_passing_neighbors, detail::kAutoTwoHopMinPassingNeighbors);
  EXPECT_EQ(
    detail::choose_filter_expansion(res, index->graph(), index->source_indices(), view).mode,
    cagra::filter_expansion_mode::PASSING_TWO_HOP);
}

// A filter that passes each row with probability 1/10, independently of its cluster: a passing
// row has about graph_degree / 10 passing neighbors, and AUTO selects SCORE_ALL.
TEST_F(CagraFilterTwoHopTest, AutoChoosesScoreAllForUniformFilter)
{
  std::mt19937 gen(42);
  std::vector<uint32_t> words((n_rows + 31) / 32, 0u);
  for (int64_t i = 0; i < n_rows; i++) {
    if (gen() % 10 == 0) { words[i / 32] |= 1u << (i % 32); }
  }
  int64_t passing = 0;
  for (uint32_t w : words) {
    passing += __builtin_popcount(w);
  }
  const int f     = add_filter(std::move(words));
  const auto view = filter(f).bitset_view_;
  const auto estimate =
    detail::estimate_passing_neighbors(res, index->graph(), index->source_indices(), view, true);
  EXPECT_GE(estimate.samples, detail::kFilterExpansionMinSamples);
  EXPECT_LT(estimate.mean_passing_neighbors, detail::kAutoTwoHopMinPassingNeighbors);
  // The same launch counts the passing rows exactly, for the filtering rate of SCORE_ALL.
  EXPECT_EQ(estimate.passing_rows, passing);
  EXPECT_EQ(
    detail::choose_filter_expansion(res, index->graph(), index->source_indices(), view).mode,
    cagra::filter_expansion_mode::SCORE_ALL);
}

// With too few passing rows to sample, AUTO selects SCORE_ALL.
TEST_F(CagraFilterTwoHopTest, AutoChoosesScoreAllForNearlyEmptyFilter)
{
  std::vector<uint32_t> words((n_rows + 31) / 32, 0u);
  words[0]    = 1u;
  const int f = add_filter(std::move(words));
  EXPECT_EQ(detail::choose_filter_expansion(
              res, index->graph(), index->source_indices(), filter(f).bitset_view_)
              .mode,
            cagra::filter_expansion_mode::SCORE_ALL);
}

TEST_F(CagraFilterTwoHopTest, AutoFindsFilteredNeighbors)
{
  auto params             = two_hop_params();
  params.algo             = cagra::search_algo::AUTO;
  params.filter_expansion = cagra::filter_expansion_mode::AUTO;
  const auto found        = search(params, 0);
  EXPECT_GE(filtered_recall(found, [](int64_t) { return 0; }), min_recall_two_hop);
}

TEST_F(CagraFilterTwoHopTest, AutoRejectsQueryBitsets)
{
  auto pointers           = query_pointers([](int64_t q) { return static_cast<int>(q % 2); });
  auto params             = two_hop_params();
  params.filter_expansion = cagra::filter_expansion_mode::AUTO;
  params.query_bitsets    = as_query_bitsets(pointers);
  EXPECT_THROW(search(params, 0), raft::exception);
}

// In the second hop, the number of graph entries a thread reads per round depends on the thread
// block size, and the distances to the child nodes are computed by teams of team_size threads.
// The test sweeps both parameters.
TEST_F(CagraFilterTwoHopTest, ThreadBlockAndTeamSizesFindFilteredNeighbors)
{
  for (size_t thread_block_size : {64, 128, 256, 512, 1024}) {
    for (size_t team_size : {0, 8, 16, 32}) {
      auto params              = two_hop_params();
      params.thread_block_size = thread_block_size;
      params.team_size         = team_size;
      const auto found         = search(params, 0);
      EXPECT_GE(filtered_recall(found, [](int64_t) { return 0; }), min_recall_two_hop)
        << "thread_block_size " << thread_block_size << ", team_size " << team_size;
    }
  }
}

TEST_F(CagraFilterTwoHopTest, QueryBitsetsFilterEachQuery)
{
  // Even queries use filter 0 and odd queries use filter 1. The bitset of the filter object
  // (filter 0) is not read.
  auto filter_of       = [](int64_t q) { return static_cast<int>(q % 2); };
  auto pointers        = query_pointers(filter_of);
  auto params          = two_hop_params();
  params.query_bitsets = as_query_bitsets(pointers);
  for (auto mode :
       {cagra::filter_expansion_mode::PASSING_TWO_HOP, cagra::filter_expansion_mode::SCORE_ALL}) {
    params.filter_expansion = mode;
    params.filtering_rate   = known_filtering_rate;
    params.max_iterations   = mode == cagra::filter_expansion_mode::SCORE_ALL ? 0 : 20;
    const auto found        = search(params, 0);
    const double recall     = filtered_recall(found, filter_of);
    EXPECT_GE(
      recall,
      mode == cagra::filter_expansion_mode::SCORE_ALL ? min_recall_score_all : min_recall_two_hop);
  }
}

TEST_F(CagraFilterTwoHopTest, QueryBitsetsFollowBatchOffset) { one_bitset_per_query_in_batches(); }

TEST_F(CagraFilterTwoHopOddWidthTest, QueryBitsetsFollowBatchOffset)
{
  one_bitset_per_query_in_batches();
}

// Only MULTI_CTA reads query_bitsets, so a search with another algorithm must be rejected.
TEST_F(CagraFilterTwoHopTest, QueryBitsetsNeedMultiCta)
{
  auto pointers        = query_pointers([](int64_t q) { return static_cast<int>(q % 2); });
  auto params          = two_hop_params();
  params.algo          = cagra::search_algo::SINGLE_CTA;
  params.query_bitsets = as_query_bitsets(pointers);
  EXPECT_THROW(search(params, 0), raft::exception);
}

// With SCORE_ALL, itopk_size is increased by the filtering rate. With query_bitsets the rate
// cannot be computed from the bitset of the filter object, so the search must be rejected unless
// filtering_rate is set.
TEST_F(CagraFilterTwoHopTest, QueryBitsetsNeedFilteringRateForScoreAll)
{
  auto pointers           = query_pointers([](int64_t q) { return static_cast<int>(q % 2); });
  auto params             = two_hop_params();
  params.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
  params.query_bitsets    = as_query_bitsets(pointers);
  EXPECT_THROW(search(params, 0), raft::logic_error);
}

// query_bitsets replaces the bitset of a bitset_filter with standard 32-bit packing, so the search
// must be given such a filter.
TEST_F(CagraFilterTwoHopTest, QueryBitsetsNeedBitsetFilter)
{
  auto pointers        = query_pointers([](int64_t q) { return static_cast<int>(q % 2); });
  auto params          = two_hop_params();
  params.query_bitsets = as_query_bitsets(pointers);

  // No filter. With algo = AUTO the search must be rejected too, whichever algorithm is selected.
  cuvs::neighbors::filtering::none_sample_filter no_filter;
  EXPECT_THROW(search(params, no_filter), raft::logic_error);
  params.algo = cagra::search_algo::AUTO;
  EXPECT_THROW(search(params, no_filter), raft::exception);
  params.algo = cagra::search_algo::MULTI_CTA;

  // A filter of another type.
  cuvs::neighbors::filtering::udf_filter udf_filter(
    "__device__ bool cuvs_filter_udf(uint32_t, source_index_t, void*) { return true; }");
  EXPECT_THROW(search(params, udf_filter), raft::logic_error);

  // A bitset_filter without standard 32-bit packing: its bitset is declared with 8-bit words.
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> bytes_filter(
    cuvs::core::bitset_view<uint32_t, int64_t>(bits[0].data_handle(), n_rows, 8));
  EXPECT_THROW(search(params, bytes_filter), raft::logic_error);
}

// Only six rows, all in one cluster, pass the filter: fewer than k.
TEST_F(CagraFilterTwoHopTest, FewerPassingRowsThanKReturnsAllOfThem)
{
  const std::vector<uint32_t> rows = {4203, 4217, 4230, 4244, 4268, 4291};
  std::vector<uint32_t> words((n_rows + 31) / 32, 0u);
  for (auto row : rows) {
    words[row / 32] |= 1u << (row % 32);
  }
  const auto found = search(two_hop_params(), add_filter(std::move(words)));
  for (int64_t q = 0; q < n_queries; q++) {
    // The first rows.size() slots must hold exactly the passing rows and the other slots must be
    // empty. The rows are sorted before the comparison, so their order is not checked.
    std::vector<uint32_t> returned(found.begin() + q * k, found.begin() + q * k + rows.size());
    std::sort(returned.begin(), returned.end());
    EXPECT_EQ(returned, rows) << "query " << q;
    for (int64_t i = rows.size(); i < k; i++) {
      EXPECT_EQ(found[q * k + i], invalid_row) << "query " << q << ", neighbor " << i;
    }
  }
}

TEST_F(CagraFilterTwoHopTest, NoPassingRowReturnsNoNeighbors)
{
  const int f      = add_filter(std::vector<uint32_t>((n_rows + 31) / 32, 0u));
  const auto found = search(two_hop_params(), f);
  EXPECT_EQ(std::count(found.begin(), found.end(), invalid_row), n_queries * k);
}

// For a filter that is not a bitset with standard 32-bit packing, PASSING_TWO_HOP calls the
// linked filter for every node it examines and selects its seed nodes by testing random nodes with
// it.
TEST_F(CagraFilterTwoHopTest, BloomFilterFindsFilteredNeighbors)
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

  // A bloom filter over the rows of filter 0. Its target false-positive rate (1e-6) is low enough
  // that no false positive is expected: one would be a passing row surrounded by filtered-out
  // rows, which PASSING_TWO_HOP may not reach.
  cuvs::core::bloom_filter bloom(
    res, n_rows, static_cast<float>(host_added.size()) / static_cast<float>(n_rows), 1e-6f);
  bloom.add(res, raft::make_const_mdspan(added.view()));

  // Build the reference filter from what the bloom filter reports as contained, so that a false
  // positive counts as a passing row.
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
  EXPECT_GE(filtered_recall(found, [f](int64_t) { return f; }), min_recall_two_hop);
}

TEST_F(CagraFilterTwoHopTest, RoaringBitmapFilterFollowsBatchOffset)
{
  one_allowlist_per_query_in_batches();
}

TEST_F(CagraFilterTwoHopOddWidthTest, RoaringBitmapFilterFollowsBatchOffset)
{
  one_allowlist_per_query_in_batches();
}

// The words of filter 0 declared as a bitset of 8-bit words pass the same rows. The kernel does
// not read the words of such a bitset directly: it calls the linked filter.
TEST_F(CagraFilterTwoHopTest, BitsetOfBytesFindsFilteredNeighbors)
{
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> bytes_filter(
    cuvs::core::bitset_view<uint32_t, int64_t>(bits[0].data_handle(), n_rows, 8));
  const auto found = search(two_hop_params(), bytes_filter);
  EXPECT_GE(filtered_recall(found, [](int64_t) { return 0; }), min_recall_two_hop);
}

// With algo = AUTO, PASSING_TWO_HOP and query_bitsets each select MULTI_CTA. max_queries is set
// so that AUTO would otherwise select SINGLE_CTA (itopk_size <= 512 and max_queries >= twice the
// number of SMs). SINGLE_CTA rejects query_bitsets, so the second search succeeding shows that
// MULTI_CTA was selected. For the first search the evidence is indirect: SINGLE_CTA would search
// with SCORE_ALL, which is not expected to reach the PASSING_TWO_HOP recall floor on this dataset.
TEST_F(CagraFilterTwoHopTest, AutoSelectsMultiCta)
{
  auto params        = two_hop_params();
  params.algo        = cagra::search_algo::AUTO;
  params.max_queries = 2 * raft::resource::get_device_properties(res).multiProcessorCount;
  EXPECT_GE(filtered_recall(search(params, 0), [](int64_t) { return 0; }), min_recall_two_hop);

  auto filter_of          = [](int64_t q) { return static_cast<int>(q % 2); };
  auto pointers           = query_pointers(filter_of);
  params.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
  params.max_iterations   = 0;
  params.filtering_rate   = known_filtering_rate;
  params.query_bitsets    = as_query_bitsets(pointers);
  EXPECT_GE(filtered_recall(search(params, 0), filter_of), min_recall_score_all);
}

// SINGLE_CTA and MULTI_KERNEL do not implement PASSING_TWO_HOP: the search logs a warning and uses
// SCORE_ALL. Only the filter is checked here: every valid row returned must pass it.
TEST_F(CagraFilterTwoHopTest, OtherAlgorithmsReturnOnlyPassingRows)
{
  for (auto algo : {cagra::search_algo::SINGLE_CTA, cagra::search_algo::MULTI_KERNEL}) {
    auto params           = two_hop_params();
    params.algo           = algo;
    params.max_iterations = 0;
    for (auto row : search(params, 0)) {
      // These algorithms do not all write invalid_row to a result slot that they could not fill.
      if (row < n_rows) { EXPECT_TRUE(passes(0, row)) << "returned row " << row; }
    }
  }
}

}  // namespace
}  // namespace cuvs::neighbors::cagra
