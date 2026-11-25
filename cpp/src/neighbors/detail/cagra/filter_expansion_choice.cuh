/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../detail/neighbors_device_intrinsics.cuh"  // xorshift64

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <optional>

namespace cuvs::neighbors::cagra::detail {

/** Rows that pass the filter sampled by estimate_passing_neighbors(). */
inline constexpr uint32_t kFilterExpansionSamples = 256;
/** Random graph nodes a warp tests per attempt to find a sample that passes the filter (32 per
 * lane). */
inline constexpr uint32_t kFilterExpansionTries = 32;
/** Fewer samples than this (the filter passes very few rows) give no estimate. */
inline constexpr uint32_t kFilterExpansionMinSamples = 32;
/**
 * filter_expansion_mode::AUTO selects PASSING_TWO_HOP when a row that passes the filter has at
 * least this many graph neighbors that pass it, on average.
 */
inline constexpr double kAutoTwoHopMinPassingNeighbors = 8.0;

/**
 * The first `sample_blocks` blocks take one sample per warp: find a random graph node that passes
 * the filter, then count its graph neighbors that pass the filter. counts[0] receives the number
 * of samples found, counts[1] the total number of passing neighbors. The other blocks, if any,
 * count the set bits of the bitset into counts[2] (the number of rows that pass), so that a
 * search that selects SCORE_ALL needs no separate count for its filtering rate.
 */
template <typename GraphIdxT, typename SourceIdxT>
RAFT_KERNEL passing_neighbors_kernel(const GraphIdxT* __restrict__ graph,
                                     int64_t graph_rows,
                                     uint32_t graph_degree,
                                     const SourceIdxT* __restrict__ source_indices,
                                     const uint32_t* __restrict__ words,
                                     int64_t bitset_size,
                                     uint64_t seed,
                                     uint32_t num_samples,
                                     uint32_t sample_blocks,
                                     unsigned long long* __restrict__ counts)
{
  constexpr uint32_t kFullMask = 0xffffffffu;
  constexpr uint32_t kWarpSize = 32;
  const uint32_t lane          = threadIdx.x % kWarpSize;

  if (blockIdx.x >= sample_blocks) {
    const int64_t num_words = (bitset_size + 31) / 32;
    const int64_t first     = static_cast<int64_t>(blockIdx.x - sample_blocks) * blockDim.x;
    const int64_t stride    = static_cast<int64_t>(gridDim.x - sample_blocks) * blockDim.x;
    unsigned long long bits = 0;
    for (int64_t w = first + threadIdx.x; w < num_words; w += stride) {
      uint32_t word = words[w];
      if (w == num_words - 1 && (bitset_size % 32) != 0) {
        word &= (1u << (bitset_size % 32)) - 1u;
      }
      bits += __popc(word);
    }
    for (uint32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
      bits += __shfl_down_sync(kFullMask, bits, offset);
    }
    if (lane == 0 && bits != 0) { atomicAdd(counts + 2, bits); }
    return;
  }
  const uint64_t sample =
    (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  if (sample >= num_samples) { return; }

  // The bitset is indexed by source index, the graph by node id.
  auto passes = [&](uint64_t node) {
    const int64_t source =
      source_indices != nullptr ? static_cast<int64_t>(source_indices[node]) : int64_t(node);
    return source >= 0 && source < bitset_size && ((words[source >> 5] >> (source & 31)) & 1u);
  };

  uint64_t r  = seed ^ (sample * 0x9e3779b97f4a7c15ull) ^ ((lane + 1) * 0xbf58476d1ce4e5b9ull);
  int64_t row = -1;
  for (uint32_t t = 0; t < kFilterExpansionTries && row < 0; t++) {
    r                   = cuvs::neighbors::detail::device::xorshift64(r);
    const uint64_t node = r % static_cast<uint64_t>(graph_rows);
    const uint32_t mask = __ballot_sync(kFullMask, passes(node));
    if (mask != 0) { row = __shfl_sync(kFullMask, static_cast<long long>(node), __ffs(mask) - 1); }
  }
  if (row < 0) { return; }

  uint32_t passing = 0;
  for (uint32_t j = lane; j < graph_degree; j += kWarpSize) {
    const GraphIdxT neighbor = graph[row * graph_degree + j];
    if (static_cast<int64_t>(neighbor) < graph_rows && passes(neighbor)) { passing++; }
  }
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
    passing += __shfl_down_sync(kFullMask, passing, offset);
  }
  if (lane == 0) {
    atomicAdd(counts, 1ull);
    atomicAdd(counts + 1, static_cast<unsigned long long>(passing));
  }
}

/** Result of estimate_passing_neighbors(). */
struct passing_neighbors_estimate {
  /** Number of sampled rows that pass the filter. */
  uint32_t samples = 0;
  /** Mean number of graph neighbors that pass the filter, over the sampled rows. */
  double mean_passing_neighbors = 0.0;
  /** Number of rows that pass the filter, or -1 when they were not counted. */
  int64_t passing_rows = -1;
};

/**
 * Estimate how many graph neighbors of a row that passes the filter also pass it, from
 * `num_samples` random graph nodes that pass the filter. The bitset is indexed by source index
 * when the index has source indices. With `count_passing_rows`, the same launch also counts the
 * rows that pass. Synchronizes the stream of `res` once.
 */
template <typename GraphIdxT, typename SourceIdxT>
passing_neighbors_estimate estimate_passing_neighbors(
  raft::resources const& res,
  raft::device_matrix_view<const GraphIdxT, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIdxT, int64_t>> source_indices,
  cuvs::core::bitset_view<uint32_t, int64_t> bitset,
  bool count_passing_rows = false,
  uint32_t num_samples    = kFilterExpansionSamples)
{
  passing_neighbors_estimate estimate{};
  if (graph.extent(0) == 0 || graph.extent(1) == 0 || bitset.data() == nullptr ||
      num_samples == 0) {
    return estimate;
  }
  auto stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<unsigned long long> counts(
    3, stream, raft::resource::get_workspace_resource_ref(res));
  RAFT_CUDA_TRY(cudaMemsetAsync(counts.data(), 0, 3 * sizeof(unsigned long long), stream.get()));

  constexpr uint32_t kBlockSize    = 256;
  const uint32_t samples_per_block = kBlockSize / 32;
  const uint32_t sample_blocks     = (num_samples + samples_per_block - 1) / samples_per_block;
  const uint32_t count_blocks      = count_passing_rows ? 128 : 0;
  const uint32_t grid              = sample_blocks + count_blocks;
  passing_neighbors_kernel<GraphIdxT, SourceIdxT><<<grid, kBlockSize, 0, stream.get()>>>(
    graph.data_handle(),
    graph.extent(0),
    static_cast<uint32_t>(graph.extent(1)),
    source_indices.has_value() ? source_indices->data_handle() : nullptr,
    bitset.data(),
    bitset.size(),
    0x5eed5eed5eed5eedull,
    num_samples,
    sample_blocks,
    counts.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  unsigned long long host_counts[3] = {0, 0, 0};
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    host_counts, counts.data(), sizeof(host_counts), cudaMemcpyDeviceToHost, stream.get()));
  raft::resource::sync_stream(res);
  estimate.samples = static_cast<uint32_t>(host_counts[0]);
  if (count_passing_rows) { estimate.passing_rows = static_cast<int64_t>(host_counts[2]); }
  if (estimate.samples > 0) {
    estimate.mean_passing_neighbors =
      static_cast<double>(host_counts[1]) / static_cast<double>(estimate.samples);
  }
  return estimate;
}

/** Result of choose_filter_expansion(). */
struct filter_expansion_choice {
  filter_expansion_mode mode = filter_expansion_mode::SCORE_ALL;
  /** Number of rows that pass the filter (counted in the same launch), or -1. */
  int64_t passing_rows = -1;
};

/**
 * Resolve filter_expansion_mode::AUTO for a bitset filter: PASSING_TWO_HOP when the rows that
 * pass the filter are close to each other in the graph (a passing row has at least
 * kAutoTwoHopMinPassingNeighbors passing graph neighbors on average), SCORE_ALL otherwise, and
 * SCORE_ALL when the estimate cannot be made (a bitset without standard 32-bit packing, or too
 * few rows that pass). With `count_passing_rows`, also returns the number of rows that pass, for
 * the filtering rate of SCORE_ALL.
 */
template <typename GraphIdxT, typename SourceIdxT>
filter_expansion_choice choose_filter_expansion(
  raft::resources const& res,
  raft::device_matrix_view<const GraphIdxT, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIdxT, int64_t>> source_indices,
  cuvs::core::bitset_view<uint32_t, int64_t> bitset,
  bool count_passing_rows = false)
{
  const bool standard_packing =
    bitset.get_original_nbits() == 0 || bitset.get_original_nbits() == 32;
  if (!standard_packing) {
    RAFT_LOG_DEBUG("filter_expansion AUTO: bitset without standard 32-bit packing, SCORE_ALL");
    return {};
  }
  const auto estimate =
    estimate_passing_neighbors(res, graph, source_indices, bitset, count_passing_rows);
  filter_expansion_choice choice{filter_expansion_mode::SCORE_ALL, estimate.passing_rows};
  if (estimate.samples < kFilterExpansionMinSamples) {
    RAFT_LOG_DEBUG("filter_expansion AUTO: %u passing rows sampled, SCORE_ALL", estimate.samples);
    return choice;
  }
  const bool two_hop = estimate.mean_passing_neighbors >= kAutoTwoHopMinPassingNeighbors;
  RAFT_LOG_DEBUG("filter_expansion AUTO: %.1f passing neighbors per passing row (%u samples), %s",
                 estimate.mean_passing_neighbors,
                 estimate.samples,
                 two_hop ? "PASSING_TWO_HOP" : "SCORE_ALL");
  if (two_hop) { choice.mode = filter_expansion_mode::PASSING_TWO_HOP; }
  return choice;
}

}  // namespace cuvs::neighbors::cagra::detail
