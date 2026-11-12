/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

// Clustered rows, and filters that pass whole clusters: the case PASSING_TWO_HOP is for.
constexpr int64_t n_rows      = 20000;
constexpr int64_t n_dim       = 16;
constexpr int64_t per_cluster = 100;
constexpr int64_t n_queries   = 32;
constexpr int64_t k           = 10;

class CagraFilterTwoHopTest : public ::testing::Test {
 protected:
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
    // Filter f passes the clusters whose number is f modulo 10: a tenth of the rows.
    for (int f = 0; f < 2; f++) {
      host_bits[f].assign((n_rows + 31) / 32, 0u);
      for (int64_t i = 0; i < n_rows; i++) {
        if ((i / per_cluster) % 10 == 3 + 4 * f) { host_bits[f][i / 32] |= 1u << (i % 32); }
      }
    }

    auto stream = raft::resource::get_cuda_stream(res);
    dataset.emplace(raft::make_device_matrix<float, int64_t>(res, n_rows, n_dim));
    queries.emplace(raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim));
    raft::copy(dataset->data_handle(), host_rows.data(), host_rows.size(), stream);
    raft::copy(queries->data_handle(), host_queries.data(), host_queries.size(), stream);
    for (int f = 0; f < 2; f++) {
      bits[f].emplace(raft::make_device_vector<uint32_t, int64_t>(res, host_bits[f].size()));
      raft::copy(bits[f]->data_handle(), host_bits[f].data(), host_bits[f].size(), stream);
    }

    cagra::index_params index_params;
    index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
    index.emplace(cagra::build(res,
                               index_params,
                               cuvs::neighbors::make_device_padded_dataset_view(
                                 res, raft::make_const_mdspan(dataset->view()))));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index.reset();
    bits[0].reset();
    bits[1].reset();
    queries.reset();
    dataset.reset();
    raft::resource::sync_stream(res);
  }

  bool passes(int f, uint32_t row) const
  {
    return row < n_rows && ((host_bits[f][row / 32] >> (row % 32)) & 1u);
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
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end());
    std::set<uint32_t> top;
    for (int64_t i = 0; i < k; i++) {
      top.insert(scored[i].second);
    }
    return top;
  }

  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(int f)
  {
    return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      cuvs::core::bitset_view<uint32_t, int64_t>(bits[f]->data_handle(), n_rows));
  }

  std::vector<uint32_t> search(cagra::search_params const& params, int f)
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
    cagra::search(res,
                  params,
                  *index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  filter(f));
    std::vector<uint32_t> out(n_queries * k);
    raft::copy(out.data(), neighbors.data_handle(), out.size(), raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return out;
  }

  // Every returned row passes its query's filter; returns the share of the exact answers found.
  template <typename FilterOf>
  double check(std::vector<uint32_t> const& found, FilterOf filter_of)
  {
    int64_t hits = 0;
    for (int64_t q = 0; q < n_queries; q++) {
      const int f      = filter_of(q);
      const auto truth = exact(f, q);
      for (int64_t i = 0; i < k; i++) {
        const uint32_t row = found[q * k + i];
        if (row == 0xffffffffu) { continue; }
        EXPECT_TRUE(passes(f, row)) << "query " << q << " returned row " << row;
        hits += truth.count(row);
      }
    }
    return static_cast<double>(hits) / static_cast<double>(n_queries * k);
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

  raft::device_resources res;
  std::vector<float> host_rows, host_queries;
  std::vector<uint32_t> host_bits[2];
  std::optional<raft::device_matrix<float, int64_t>> dataset, queries;
  std::optional<raft::device_vector<uint32_t, int64_t>> bits[2];
  std::optional<cagra::index<float, uint32_t>> index;
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

TEST_F(CagraFilterTwoHopTest, OneBitsetPerQuery)
{
  // Even queries take filter 0, odd ones filter 1; the filter object's own bitset is not read.
  std::vector<uintptr_t> host_pointers(n_queries);
  for (int64_t q = 0; q < n_queries; q++) {
    host_pointers[q] = reinterpret_cast<uintptr_t>(bits[q % 2]->data_handle());
  }
  auto pointers = raft::make_device_vector<uintptr_t, int64_t>(res, n_queries);
  raft::copy(pointers.data_handle(),
             host_pointers.data(),
             host_pointers.size(),
             raft::resource::get_cuda_stream(res));

  auto params          = two_hop_params();
  params.query_bitsets = reinterpret_cast<const uint32_t* const*>(pointers.data_handle());
  for (auto mode :
       {cagra::filter_expansion_mode::PASSING_TWO_HOP, cagra::filter_expansion_mode::SCORE_ALL}) {
    params.filter_expansion = mode;
    params.filtering_rate   = 0.9f;
    params.max_iterations   = mode == cagra::filter_expansion_mode::SCORE_ALL ? 0 : 20;
    const auto found        = search(params, 0);
    const double recall     = check(found, [](int64_t q) { return static_cast<int>(q % 2); });
    EXPECT_GE(recall, mode == cagra::filter_expansion_mode::SCORE_ALL ? 0.5 : 0.9);
  }
}

TEST_F(CagraFilterTwoHopTest, QueryBitsetsNeedMultiCta)
{
  auto pointers = raft::make_device_vector<uintptr_t, int64_t>(res, n_queries);
  auto params   = two_hop_params();
  params.algo   = cagra::search_algo::SINGLE_CTA;
  params.query_bitsets = reinterpret_cast<const uint32_t* const*>(pointers.data_handle());
  EXPECT_ANY_THROW(search(params, 0));
}

}  // namespace
}  // namespace cuvs::neighbors::cagra
