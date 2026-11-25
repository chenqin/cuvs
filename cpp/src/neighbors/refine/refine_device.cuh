/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_flat/ivf_flat_build.cuh"
#include "../ivf_flat/ivf_flat_interleaved_scan_ext.cuh"
#include "refine_common.hpp"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/selection/select_k.hpp>
#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/init.cuh>

#include <cstdint>
#include <limits>

namespace cuvs::neighbors {

namespace detail {

/**
 * One warp per candidate: the distance between query `q` and its candidate `c`, written to
 * `distances[q * n_candidates + c]`. A candidate id outside the dataset gets `worst`.
 */
template <typename data_t, typename idx_t, typename distance_t>
RAFT_KERNEL refine_candidate_distances_kernel(const data_t* __restrict__ dataset,
                                              int64_t n_rows,
                                              int64_t dim,
                                              const data_t* __restrict__ queries,
                                              const idx_t* __restrict__ candidates,
                                              int64_t n_queries,
                                              int64_t n_candidates,
                                              cuvs::distance::DistanceType metric,
                                              distance_t worst,
                                              distance_t* __restrict__ distances)
{
  constexpr uint32_t kWarpSize = 32;
  const int64_t pair  = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const uint32_t lane = threadIdx.x % kWarpSize;
  if (pair >= n_queries * n_candidates) { return; }
  const int64_t q  = pair / n_candidates;
  const auto id    = candidates[pair];
  const bool valid = static_cast<int64_t>(id) >= 0 && static_cast<int64_t>(id) < n_rows;

  const bool l2 = metric == cuvs::distance::DistanceType::L2Expanded ||
                  metric == cuvs::distance::DistanceType::L2Unexpanded;
  float acc = 0, query_norm = 0, row_norm = 0;
  if (valid) {
    const data_t* x = dataset + static_cast<int64_t>(id) * dim;
    const data_t* y = queries + q * dim;
    for (int64_t j = lane; j < dim; j += kWarpSize) {
      const float a = static_cast<float>(y[j]);
      const float b = static_cast<float>(x[j]);
      if (l2) {
        acc += (a - b) * (a - b);
      } else {
        acc += a * b;
        query_norm += a * a;
        row_norm += b * b;
      }
    }
  }
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
    query_norm += __shfl_down_sync(0xffffffffu, query_norm, offset);
    row_norm += __shfl_down_sync(0xffffffffu, row_norm, offset);
  }
  if (lane != 0) { return; }
  distance_t d = worst;
  if (valid) {
    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      const float denom = sqrtf(query_norm) * sqrtf(row_norm);
      d                 = static_cast<distance_t>(denom > 0 ? 1.0f - acc / denom : 1.0f);
    } else {
      d = static_cast<distance_t>(acc);
    }
  }
  distances[pair] = d;
}

/**
 * Refinement for k larger than the capacity of the IVF-Flat scan's per-block top-k: compute the
 * distance of every candidate, then select the best k with cuvs::selection::select_k.
 * Distances follow refine_host: squared L2 for L2Expanded and L2Unexpanded, the inner product for
 * InnerProduct (larger is better), and 1 - cosine similarity for CosineExpanded.
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_device_all_candidates(
  raft::resources const& handle,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric)
{
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2Unexpanded ||
                 metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "refine with k > %d supports L2Expanded, L2Unexpanded, InnerProduct and "
               "CosineExpanded",
               raft::matrix::detail::select::warpsort::kMaxCapacity);
  const int64_t n_queries    = queries.extent(0);
  const int64_t n_candidates = neighbor_candidates.extent(1);
  const int64_t k            = indices.extent(1);
  const bool select_min      = metric != cuvs::distance::DistanceType::InnerProduct;
  const distance_t worst =
    select_min ? std::numeric_limits<distance_t>::max() : std::numeric_limits<distance_t>::lowest();
  auto stream = raft::resource::get_cuda_stream(handle);

  auto all =
    raft::make_device_mdarray<distance_t>(handle,
                                          raft::resource::get_workspace_resource_ref(handle),
                                          raft::make_extents<int64_t>(n_queries, n_candidates));
  constexpr uint32_t kBlockSize = 256;
  const int64_t pairs           = n_queries * n_candidates;
  const int64_t blocks          = (pairs * 32 + kBlockSize - 1) / kBlockSize;
  if (pairs > 0) {
    refine_candidate_distances_kernel<data_t, idx_t, distance_t>
      <<<static_cast<uint32_t>(blocks), kBlockSize, 0, stream.get()>>>(
        dataset.data_handle(),
        dataset.extent(0),
        dataset.extent(1),
        queries.data_handle(),
        neighbor_candidates.data_handle(),
        n_queries,
        n_candidates,
        metric,
        worst,
        all.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  cuvs::selection::select_k(
    handle,
    raft::make_device_matrix_view<const distance_t, int64_t>(
      all.data_handle(), n_queries, n_candidates),
    std::make_optional(raft::make_device_matrix_view<const idx_t, int64_t>(
      neighbor_candidates.data_handle(), n_queries, n_candidates)),
    raft::make_device_matrix_view<distance_t, int64_t>(distances.data_handle(), n_queries, k),
    raft::make_device_matrix_view<idx_t, int64_t>(indices.data_handle(), n_queries, k),
    select_min,
    true);
}

/**
 * See cuvs::neighbors::refine for docs.
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_device(
  raft::resources const& handle,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  matrix_idx n_candidates = neighbor_candidates.extent(1);
  matrix_idx n_queries    = queries.extent(0);
  matrix_idx dim          = dataset.extent(1);
  uint32_t k              = static_cast<uint32_t>(indices.extent(1));

  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "neighbors::refine(%zu, %u)", size_t(n_queries), uint32_t(n_candidates));

  refine_check_input(dataset.extents(),
                     queries.extents(),
                     neighbor_candidates.extents(),
                     indices.extents(),
                     distances.extents(),
                     metric);

  // The IVF-Flat scan below keeps a per-block top-k of at most kMaxCapacity entries. For larger k,
  // compute the distance of every candidate and select the best k.
  if (k > raft::matrix::detail::select::warpsort::kMaxCapacity) {
    return refine_device_all_candidates<idx_t, data_t, distance_t, matrix_idx>(
      handle, dataset, queries, neighbor_candidates, indices, distances, metric);
  }

  // The refinement search can be mapped to an IVF flat search:
  // - We consider that the candidate vectors form a cluster, separately for each query.
  // - In other words, the n_queries * n_candidates vectors form n_queries clusters, each with
  //   n_candidates elements.
  // - We consider that the coarse level search is already performed and assigned a single cluster
  //   to search for each query (the cluster formed from the corresponding candidates).
  // - We run IVF flat search with n_probes=1 to select the best k elements of the candidates.
  rmm::device_uvector<uint32_t> fake_coarse_idx(n_queries, raft::resource::get_cuda_stream(handle));

  raft::linalg::map_offset(handle,
                           raft::make_device_vector_view(fake_coarse_idx.data(), n_queries),
                           raft::cast_op<uint32_t>{});

  cuvs::neighbors::ivf_flat::index<data_t, int64_t> refinement_index(
    handle, metric, n_queries, false, true, dim);

  cuvs::neighbors::ivf_flat::detail::fill_refinement_index<data_t, int64_t>(
    handle,
    &refinement_index,
    dataset.data_handle(),
    neighbor_candidates.data_handle(),
    static_cast<int64_t>(n_queries),
    static_cast<uint32_t>(n_candidates));
  uint32_t grid_dim_x = 1;

  // the neighbor ids will be computed in uint32_t as offset
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, raft::resource::get_cuda_stream(handle));
  // Offsets per probe for each query [n_queries] as n_probes = 1
  rmm::device_uvector<uint32_t> chunk_index(n_queries, raft::resource::get_cuda_stream(handle));

  // we know that each cluster has exactly n_candidates entries
  raft::matrix::fill(
    handle, raft::make_device_vector_view(chunk_index.data(), n_queries), uint32_t(n_candidates));

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(idx_t) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(indices.data_handle());
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k),
                                raft::resource::get_cuda_stream(handle));
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  using acc_t = typename cuvs::spatial::knn::detail::utils::config<data_t>::value_t;

  cuvs::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<data_t, acc_t, int64_t>(
    refinement_index,
    queries.data_handle(),
    fake_coarse_idx.data(),
    static_cast<uint32_t>(n_queries),
    0,
    metric,
    1,
    k,
    0,
    chunk_index.data(),
    cuvs::distance::is_min_close(metric),
    cuvs::neighbors::filtering::none_sample_filter(),
    neighbors_uint32,
    distances.data_handle(),
    grid_dim_x,
    raft::resource::get_cuda_stream(handle),
    std::nullopt);

  // postprocessing -- neighbors from position to actual id
  cuvs::neighbors::ivf::detail::postprocess_neighbors(indices.data_handle(),
                                                      neighbors_uint32,
                                                      refinement_index.inds_ptrs().data_handle(),
                                                      fake_coarse_idx.data(),
                                                      chunk_index.data(),
                                                      n_queries,
                                                      1,
                                                      k,
                                                      raft::resource::get_cuda_stream(handle));
}

}  // namespace detail

template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_impl(
  raft::resources const& handle,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  detail::refine_device(handle, dataset, queries, neighbor_candidates, indices, distances, metric);
}
}  // namespace cuvs::neighbors
