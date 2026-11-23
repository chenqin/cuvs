/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

#include <cuda_fp16.h>

#include <random>
#include <string>
#include <vector>

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, half, std::uint32_t> AnnCagraTestF16_U32;
TEST_P(AnnCagraTestF16_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF16_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraAddNodesTest<float, half, std::uint32_t> AnnCagraAddNodesTestF16_U32;
TEST_P(AnnCagraAddNodesTestF16_U32, AnnCagraAddNodes) { this->testCagra(); }

typedef AnnCagraIndexMergeTest<float, half, std::uint32_t> AnnCagraIndexMergeTestF16_U32;
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_I64) { this->testCagra<int64_t>(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF16_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestF16_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF16_U32,
                        ::testing::ValuesIn(inputs));

typedef AnnCagraMultiPartitionTest<float, half, std::uint32_t> AnnCagraMultiPartitionTestF16_U32;
TEST_P(AnnCagraMultiPartitionTestF16_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestF16_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestF16_U32,
                        ::testing::ValuesIn(inputs_mp));

// The FP8 filtered scan searches one index, and multi-partition search walks every partition's
// graph: it refuses SCAN, scan_retry and a partition without a graph. (It takes one bitset per
// partition rather than a filter, so a row_list_filter cannot reach it.)
TEST(AnnCagraMultiPartitionF16, RejectsFilteredScan)
{
  raft::resources res;
  constexpr int64_t part_rows = 1000, dim = 16, n_queries = 4, k = 4;
  auto stream = raft::resource::get_cuda_stream(res);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::vector<half> host_rows(2 * part_rows * dim);
  for (auto& x : host_rows) {
    x = __float2half(uniform(gen));
  }
  auto rows = raft::make_device_matrix<half, int64_t>(res, 2 * part_rows, dim);
  raft::copy(rows.data_handle(), host_rows.data(), host_rows.size(), stream);

  cagra::index_params index_params;
  index_params.graph_degree              = 32;
  index_params.intermediate_graph_degree = 64;
  index_params.graph_build_params =
    graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
  std::vector<cuvs::neighbors::test::padded_device_matrix_for_cagra<half>> padded;
  std::vector<cagra::index<half, uint32_t>> parts;
  padded.reserve(2);
  parts.reserve(2);
  for (int64_t p = 0; p < 2; p++) {
    padded.emplace_back(res,
                        raft::make_device_matrix_view<const half, int64_t>(
                          rows.data_handle() + p * part_rows * dim, part_rows, dim));
    parts.push_back(cagra::build(res, index_params, padded.back().view));
  }

  auto queries =
    raft::make_device_matrix_view<const half, int64_t>(rows.data_handle(), n_queries, dim);
  auto partition_ids = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
  auto neighbors     = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
  auto distances     = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
  auto search_parts  = [&](cagra::search_params const& params,
                          std::vector<const cagra::index<half, uint32_t>*> const& indices) {
    cagra::search(
      res, params, indices, queries, partition_ids.view(), neighbors.view(), distances.view());
    raft::resource::sync_stream(res);
  };
  const std::vector<const cagra::index<half, uint32_t>*> both{&parts[0], &parts[1]};

  cagra::search_params search_params;
  EXPECT_NO_THROW(search_parts(search_params, both));

  auto scan           = search_params;
  scan.filtered_route = filtered_route_mode::SCAN;
  EXPECT_THROW(search_parts(scan, both), raft::logic_error);

  // The flags are not read: the search is refused first.
  auto retry            = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto with_retry       = search_params;
  with_retry.scan_retry = retry.data_handle();
  EXPECT_THROW(search_parts(with_retry, both), raft::logic_error);

  // The second partition's rows without its graph. The partitions' graph degrees differ too, which
  // is refused with another message, so check that this refusal names the missing graph.
  cagra::index<half, uint32_t> without_graph(res, cuvs::distance::DistanceType::L2Expanded);
  without_graph = cagra::update_dataset(res, std::move(without_graph), padded[1].view);
  try {
    search_parts(search_params, {&parts[0], &without_graph});
    ADD_FAILURE() << "a partition without a graph was searched";
  } catch (const raft::logic_error& e) {
    EXPECT_NE(std::string(e.what()).find("no graph"), std::string::npos) << e.what();
  }
}

}  // namespace cuvs::neighbors::cagra
