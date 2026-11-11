/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../hashmap.hpp"
#include "../utils.hpp"

#include <neighbors/detail/cagra/multi_partition_desc.hpp>
#include <neighbors/detail/cagra/search_multi_cta_helpers.cuh>

#include <raft/util/integer_utils.hpp>

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

#ifdef _CLK_BREAKDOWN
#include <cstdio>
#endif

#include "cagra_filter_payload.cuh"
#include "device_common_jit.cuh"
#include "extern_device_functions.cuh"

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

using cuvs::neighbors::cagra::detail::device::compute_distance_to_child_nodes_jit;
using cuvs::neighbors::cagra::detail::device::compute_distance_to_random_nodes_jit;
using cuvs::neighbors::detail::sample_filter;

// Start nodes of a PASSING_TWO_HOP search: `num_pickup` slots, each the nearest of
// `num_distilation` random nodes that pass the filter. With the filter's words at hand
// (`filter_words`, indexed by node id) a draw reads random words and takes a set bit of the first
// that has one, so it costs a few loads however few rows pass; otherwise it tests random nodes
// one by one with `passes`. A slot whose draws all fail stays empty.
template <typename IndexT, typename DistanceT, typename DataT, typename PassT>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_passing_nodes_jit(
  IndexT* __restrict__ result_indices_ptr,       // [num_pickup]
  DistanceT* __restrict__ result_distances_ptr,  // [num_pickup]
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const uint32_t num_pickup,
  const uint32_t num_distilation,
  const uint64_t rand_xor_mask,
  IndexT* __restrict__ visited_hash_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hash_ptr,
  const uint32_t traversed_hash_bitlen,
  const uint32_t block_id,
  const uint32_t num_blocks,
  const IndexT node_limit,
  const std::uint32_t* __restrict__ filter_words,
  PassT passes)
{
  constexpr IndexT invalid_index = ~static_cast<IndexT>(0);
  constexpr uint32_t kRounds     = 8;  // rounds of kWide tries a draw
  constexpr uint32_t kWide       = 4;  // words (or nodes) read together in a round

  const uint32_t team_size_bits = smem_desc->team_size_bitshift_from_smem();
  const auto args_load          = smem_desc->args.load();
  const auto max_i = raft::round_up_safe<uint32_t>(num_pickup, device::warp_size >> team_size_bits);
  const uint32_t num_words =
    static_cast<uint32_t>((static_cast<uint64_t>(node_limit) + 31) / 32);

  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += (blockDim.x >> team_size_bits)) {
    const bool valid_i = (i < num_pickup);

    IndexT best_index_team_local    = raft::upper_bound<IndexT>();
    DistanceT best_norm2_team_local = raft::upper_bound<DistanceT>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      // Every thread of a team derives the same node, which compute_distance relies on.
      IndexT seed_index = invalid_index;
      if (valid_i) {
        const uint64_t gid =
          block_id + (static_cast<uint64_t>(num_blocks) * (i + (num_pickup * j)));
        for (uint32_t round = 0; round < kRounds && seed_index == invalid_index; round++) {
          uint64_t r[kWide];
          for (uint32_t u = 0; u < kWide; u++) {
            r[u] = cuvs::neighbors::detail::device::xorshift64(
              (gid + 0x9E3779B97F4A7C15ull * (round * kWide + u + 1)) ^ rand_xor_mask);
          }
          if (filter_words != nullptr) {
            std::uint32_t word[kWide];
            for (uint32_t u = 0; u < kWide; u++) {
              word[u] = filter_words[(r[u] >> 8) % num_words];
            }
            for (uint32_t u = 0; u < kWide && seed_index == invalid_index; u++) {
              if (word[u] == 0) { continue; }
              // A set bit picked by a random rotation, so a word does not always give its lowest.
              const uint32_t rot     = static_cast<uint32_t>(r[u]) & 31u;
              const uint32_t rotated = (word[u] >> rot) | (word[u] << ((32u - rot) & 31u));
              const uint32_t bit     = (static_cast<uint32_t>(__ffs(rotated)) - 1u + rot) & 31u;
              const uint64_t node    = ((r[u] >> 8) % num_words) * 32ull + bit;
              if (node < node_limit) { seed_index = static_cast<IndexT>(node); }
            }
          } else {
            for (uint32_t u = 0; u < kWide && seed_index == invalid_index; u++) {
              const IndexT node = static_cast<IndexT>((r[u] >> 8) % node_limit);
              if (passes(node)) { seed_index = node; }
            }
          }
        }
      }
      const bool valid_seed = valid_i && (seed_index != invalid_index);

      const auto norm2 = cuvs::neighbors::cagra::detail::compute_distance<DataT, IndexT, DistanceT>(
        args_load, valid_seed ? seed_index : 0, valid_seed, team_size_bits);

      if (valid_seed && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x & ((1u << team_size_bits) - 1u);
    if (valid_i && lane_id == 0) {
      if (best_index_team_local != raft::upper_bound<IndexT>()) {
        if (hashmap::insert(visited_hash_ptr, visited_hash_bitlen, best_index_team_local) == 0) {
          // The same node was drawn for another slot.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        } else if (hashmap::search<IndexT, 1>(
                     traversed_hash_ptr, traversed_hash_bitlen, best_index_team_local)) {
          // Another block has expanded it already.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        }
      }
      result_distances_ptr[i] = best_norm2_team_local;
      result_indices_ptr[i]   = best_index_team_local;
    }
  }
}

// Children of a PASSING_TWO_HOP parent, with their distances: the parent's neighbors that pass
// the filter and, through each neighbor that does not, that neighbor's own passing neighbors.
// A rejected node is never scored and never enters the list. Children fill the buffer from
// `max_result_position` down to `min_result_position`; the first hop goes first and what does
// not fit is dropped. On return `result_position[0]` is the first slot filled.
template <typename IndexT, typename DistanceT, typename DataT, typename PassT>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_passing_child_nodes_jit(
  IndexT* __restrict__ result_indices_ptr,
  DistanceT* __restrict__ result_distances_ptr,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  IndexT* __restrict__ visited_hashmap_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hashmap_ptr,
  const uint32_t traversed_hash_bitlen,
  const IndexT parent_id,
  IndexT* __restrict__ bridges,  // [knn_k], shared memory
  int* __restrict__ result_position,
  const int min_result_position,
  const int max_result_position,
  const std::uint32_t* __restrict__ filter_words,  // indexed by node id, or nullptr
  PassT passes)
{
  constexpr IndexT invalid_index = ~static_cast<IndexT>(0);

  // A passing node becomes a child unless it is on this block's list already or some block has
  // expanded it.
  auto take = [&](const IndexT child) {
    if (hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child) == 0) { return; }
    if (hashmap::search<IndexT, 1>(traversed_hashmap_ptr, traversed_hash_bitlen, child)) { return; }
    const int j = atomicSub(result_position, 1) - 1;
    if (j >= min_result_position) { result_indices_ptr[j] = child; }
  };

  // First hop: passing neighbors are children, the others are bridges to the second hop.
  for (uint32_t i = threadIdx.x; i < knn_k; i += blockDim.x) {
    IndexT bridge = invalid_index;
    if (parent_id != invalid_index) {
      const IndexT child = knn_graph[i + (static_cast<int64_t>(knn_k) * parent_id)];
      if (child != invalid_index) {
        if (passes(child)) {
          take(child);
        } else {
          bridge = child;
        }
      }
    }
    bridges[i] = bridge;
  }
  __syncthreads();

  // Second hop: the neighbors of each bridge, until the buffer is full. A thread reads a few
  // graph entries, then their filter words, before it uses any of them, so the loads overlap;
  // a warp's threads read one bridge's row together. At most 256 entries are examined between
  // two looks at the buffer, which bounds what one round adds to the visited hash.
  const uint32_t num_slots = knn_k * knn_k;
  const uint32_t unroll    = blockDim.x >= 256 ? 1u : (256u / blockDim.x);
  constexpr uint32_t kMaxUnroll = 4;
  for (uint32_t s0 = threadIdx.x; s0 < num_slots; s0 += blockDim.x * unroll) {
    if (result_position[0] <= min_result_position) { break; }
    IndexT child[kMaxUnroll];
    for (uint32_t u = 0; u < kMaxUnroll; u++) {
      const uint32_t s    = s0 + (blockDim.x * u);
      const IndexT bridge = (u < unroll && s < num_slots) ? bridges[s / knn_k] : invalid_index;
      child[u]            = (bridge != invalid_index)
                              ? knn_graph[(s % knn_k) + (static_cast<int64_t>(knn_k) * bridge)]
                              : invalid_index;
    }
    if (filter_words != nullptr) {
      std::uint32_t word[kMaxUnroll];
      for (uint32_t u = 0; u < kMaxUnroll; u++) {
        word[u] = (child[u] != invalid_index) ? filter_words[child[u] >> 5] : 0u;
      }
      for (uint32_t u = 0; u < kMaxUnroll; u++) {
        if ((word[u] >> (child[u] & 31u)) & 1u) { take(child[u]); }
      }
    } else {
      for (uint32_t u = 0; u < kMaxUnroll; u++) {
        if (child[u] != invalid_index && passes(child[u])) { take(child[u]); }
      }
    }
  }
  __syncthreads();

  // Distances of the children, as compute_distance_to_child_nodes_jit computes them.
  const auto team_size_bits = smem_desc->team_size_bitshift_from_smem();
  const int first =
    result_position[0] < min_result_position ? min_result_position : result_position[0];
  const auto max_i = raft::round_up_safe<uint32_t>(
    static_cast<uint32_t>(max_result_position - min_result_position),
    device::warp_size >> team_size_bits);
  const auto args      = smem_desc->args.load();
  const bool lead_lane = (threadIdx.x & ((1u << team_size_bits) - 1u)) == 0;

  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += blockDim.x >> team_size_bits) {
    const int j         = static_cast<int>(i) + first;
    const bool valid_i  = (j < max_result_position);
    const auto child_id = valid_i ? result_indices_ptr[j] : invalid_index;

    const auto per_thread =
      (child_id != invalid_index)
        ? cuvs::neighbors::cagra::detail::compute_distance_per_thread<DataT, IndexT, DistanceT>(
            args, child_id)
        : (lead_lane ? raft::upper_bound<DistanceT>() : 0);
    const DistanceT child_dist = device::team_sum(per_thread, team_size_bits);
    __syncwarp();

    if (valid_i && lead_lane) { result_distances_ptr[j] = child_dist; }
  }
  // Slots below the list's end were never written.
  if (threadIdx.x == 0) { result_position[0] = first; }
}

// Shared MULTI_CTA search body for both single-partition and multi-partition kernels.
//
// The multi-partition launch is a strict generalization of the single-partition one: it adds a
// partition axis on blockIdx.z. The single-partition kernel launches a 2D grid (gridDim.z == 1,
// blockIdx.z == 0), so `row` collapses to `query_id` and `num_blocks` collapses to the 2D form.
// Per-partition differences are passed in as parameters:
//   - graph_degree:        per-partition degree used for traversal
//   - buffer_graph_degree: degree used to size result_buffer (max across partitions for mp, so the
//                          smem layout is uniform across the grid; == graph_degree
//                          single-partition)
//   - source_indices_ptr / seed_ptr / num_seeds / num_executed_iterations / graph_size:
//                          single-partition passes real values; mp passes nullptr / 0
//   - the filter payload carries this partition's own bitset (multi_partition_desc_t::bitset_ptr),
//     tested at partition-local indices
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename BitsetT = cagra_bitset<SourceIndexT>>
RAFT_DEVICE_INLINE_FUNCTION void search_core(
  IndexT* const result_indices_ptr,       // [num_queries, num_cta_per_query, itopk_size]
  DistanceT* const result_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const DataT* const queries_ptr,  // [num_queries, dataset_dim]
  const IndexT* const knn_graph,   // [dataset_size, graph_degree]
  const uint32_t max_elements,
  const uint32_t graph_degree,
  const uint32_t buffer_graph_degree,
  const SourceIndexT* source_indices_ptr,  // [num_queries, search_width]
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  const uint32_t visited_hash_bitlen,
  IndexT* const traversed_hashmap_ptr,  // [num_queries, 1 << traversed_hash_bitlen]
  const uint32_t traversed_hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations, /* stats */
  const IndexT graph_size,
  const uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  // Maps a graph node id to its source index (for output and filtering). The multi-partition bit
  // offset is NOT folded in here — it applies only to the bitset test below, not to output indices.
  auto to_source_index = [source_indices_ptr](INDEX_T x) {
    return source_indices_ptr == nullptr ? static_cast<SourceIndexT>(x) : source_indices_ptr[x];
  };

  const auto num_queries       = gridDim.y;
  const auto query_id          = blockIdx.y;
  const auto num_cta_per_query = gridDim.x;
  const auto cta_id            = blockIdx.x;  // local CTA ID
  const auto partition_id      = blockIdx.z;
  // Generalized row index: single-partition launches gridDim.z == 1 so row == query_id.
  const auto row = partition_id * num_queries + query_id;

  // The filter. When the host says it is a bitset whose words can be read as they are (see
  // cagra_sample_filter_flags), a test is one load of a word; any other filter goes through the
  // linked sample_filter. With one bitset per query, a query whose pointer is null is unfiltered.
  const bool passing_two_hop     = (filter_payload.flags & kCagraFilterPassingTwoHop) != 0;
  const bool has_query_bitsets   = (filter_payload.flags & kCagraFilterQueryBitsets) != 0;
  const std::uint32_t* filter_words = nullptr;
  if (has_query_bitsets) {
    filter_words = static_cast<const std::uint32_t* const*>(filter_payload.filter_data)
      [query_id + query_id_offset];
  } else if (filter_payload.flags & kCagraFilterBitsetWords) {
    filter_words = static_cast<const BitsetT*>(filter_payload.filter_data)->bitset_ptr;
  }
  auto passes = [&](const INDEX_T node) -> bool {
    const auto source = to_source_index(node);
    if (filter_words != nullptr) { return (filter_words[source >> 5] >> (source & 31u)) & 1u; }
    if (has_query_bitsets) { return true; }
    return sample_filter<SourceIndexT>(
      query_id + query_id_offset, source, filter_payload.sample_filter_data());
  };
  // The same words indexed by graph node id, which they are unless nodes are renumbered.
  const std::uint32_t* const node_filter_words =
    (source_indices_ptr == nullptr) ? filter_words : nullptr;

#ifdef _CLK_BREAKDOWN
  uint64_t clk_init                 = 0;
  uint64_t clk_compute_1st_distance = 0;
  uint64_t clk_topk                 = 0;
  uint64_t clk_pickup_parents       = 0;
  uint64_t clk_compute_distance     = 0;
  uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  // +----------------+---------+---------------------------+
  // | internal_top_k | padding | neighbors of parent nodes |
  // | <itopk_size>   | upto 32 | <graph_degree>            |
  // +----------------+---------+---------------------------+
  // |<---        result_buffer_size_32                 --->|
  const auto result_buffer_size    = itopk_size + buffer_graph_degree;
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  assert(result_buffer_size_32 <= max_elements);

  // Get smem_ws_size_in_bytes directly from base descriptor
  uint32_t smem_ws_size_in_bytes = dataset_desc->smem_ws_size_in_bytes();

  auto smem_desc =
    setup_workspace<DataT, IndexT, DistanceT>(dataset_desc, smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + smem_ws_size_in_bytes);
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ local_visited_hashmap_ptr =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_indices_buffer =
    reinterpret_cast<INDEX_T*>(local_visited_hashmap_ptr + hashmap::get_size(visited_hash_bitlen));
  auto* __restrict__ result_position = reinterpret_cast<int*>(parent_indices_buffer + 1);
  // PASSING_TWO_HOP only (the host adds it to the block's shared memory then): the expanded
  // node's rejected neighbors.
  auto* __restrict__ bridges = reinterpret_cast<INDEX_T*>(result_position + 1);

  INDEX_T* const local_traversed_hashmap_ptr =
    traversed_hashmap_ptr + (hashmap::get_size(traversed_hash_bitlen) * row);

  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
    result_indices_buffer[i]   = invalid_index;
    result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
  }
  hashmap::init<INDEX_T>(local_visited_hashmap_ptr, visited_hash_bitlen);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes using JIT version
  _CLK_START();
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  uint32_t block_id                   = cta_id + (num_cta_per_query * row);
  uint32_t num_blocks                 = num_cta_per_query * num_queries * gridDim.z;

  if (passing_two_hop && local_seed_ptr == nullptr) {
    compute_distance_to_random_passing_nodes_jit<IndexT, DistanceT, DataT>(
      result_indices_buffer,
      result_distances_buffer,
      smem_desc,
      graph_degree,
      num_distilation,
      rand_xor_mask,
      local_visited_hashmap_ptr,
      visited_hash_bitlen,
      local_traversed_hashmap_ptr,
      traversed_hash_bitlen,
      block_id,
      num_blocks,
      graph_size > 0 ? graph_size : smem_desc->size,
      node_filter_words,
      passes);
  } else {
    compute_distance_to_random_nodes_jit<IndexT, DistanceT, DataT>(result_indices_buffer,
                                                                   result_distances_buffer,
                                                                   smem_desc,
                                                                   graph_degree,
                                                                   num_distilation,
                                                                   rand_xor_mask,
                                                                   local_seed_ptr,
                                                                   num_seeds,
                                                                   local_visited_hashmap_ptr,
                                                                   visited_hash_bitlen,
                                                                   local_traversed_hashmap_ptr,
                                                                   traversed_hash_bitlen,
                                                                   block_id,
                                                                   num_blocks,
                                                                   graph_size);
  }
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  uint32_t iter = 0;
  while (1) {
    _CLK_START();
    if (threadIdx.x < 32) {
      // [1st warp] Topk with bitonic sort
      if constexpr (std::is_same_v<INDEX_T, uint32_t>) {
        // use a non-template wrapper function to avoid pre-inlining the topk_by_bitonic_sort
        // function (vs post-inlining, this impacts register pressure)
        if (max_elements <= 64) {
          topk_by_bitonic_sort_wrapper_64(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else if (max_elements <= 128) {
          topk_by_bitonic_sort_wrapper_128(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else {
          assert(max_elements <= 256);
          topk_by_bitonic_sort_wrapper_256(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        }
      } else {
        if (max_elements <= 64) {
          topk_by_bitonic_sort<64, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else if (max_elements <= 128) {
          topk_by_bitonic_sort<128, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else {
          assert(max_elements <= 256);
          topk_by_bitonic_sort<256, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        }
      }
    }
    __syncthreads();
    _CLK_REC(clk_topk);

    if (iter + 1 >= max_iteration) { break; }

    _CLK_START();
    if (threadIdx.x < 32) {
      // [1st warp] Pick up a next parent
      pickup_next_parent<INDEX_T, DISTANCE_T>(parent_indices_buffer,
                                              result_indices_buffer,
                                              result_distances_buffer,
                                              local_traversed_hashmap_ptr,
                                              traversed_hash_bitlen);
    } else {
      // [Other warps] Reset visited hashmap
      hashmap::init<INDEX_T>(local_visited_hashmap_ptr, visited_hash_bitlen, 32);
    }
    __syncthreads();
    _CLK_REC(clk_pickup_parents);

    if ((parent_indices_buffer[0] == invalid_index) && (iter >= min_iteration)) { break; }

    _CLK_START();
    for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index) { continue; }
      if ((i >= itopk_size) && (index & index_msb_1_mask)) {
        // Remove nodes kicked out of the itopk list from the traversed hash table.
        hashmap::remove<INDEX_T>(
          local_traversed_hashmap_ptr, traversed_hash_bitlen, index & ~index_msb_1_mask);
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      } else {
        // Restore visited hashmap by putting nodes on result buffer in it.
        index &= ~index_msb_1_mask;
        hashmap::insert(local_visited_hashmap_ptr, visited_hash_bitlen, index);
      }
    }
    // Initialize buffer for compute_distance_to_child_nodes.
    if (threadIdx.x == blockDim.x - 1) { result_position[0] = result_buffer_size_32; }
    __syncthreads();

    // Compute the norms between child nodes and query node using JIT version
    if (passing_two_hop) {
      const INDEX_T parent_slot = parent_indices_buffer[0];
      const INDEX_T parent_id   = (parent_slot != invalid_index)
                                    ? (result_indices_buffer[parent_slot] & ~index_msb_1_mask)
                                    : invalid_index;
      compute_distance_to_passing_child_nodes_jit<IndexT, DistanceT, DataT>(
        result_indices_buffer,
        result_distances_buffer,
        smem_desc,
        knn_graph,
        graph_degree,
        local_visited_hashmap_ptr,
        visited_hash_bitlen,
        local_traversed_hashmap_ptr,
        traversed_hash_bitlen,
        parent_id,
        bridges,
        result_position,
        static_cast<int>(itopk_size),
        static_cast<int>(result_buffer_size_32),
        node_filter_words,
        passes);
    } else {
      compute_distance_to_child_nodes_jit<IndexT, DistanceT, DataT, 0>(result_indices_buffer,
                                                                       result_distances_buffer,
                                                                       smem_desc,
                                                                       knn_graph,
                                                                       graph_degree,
                                                                       local_visited_hashmap_ptr,
                                                                       visited_hash_bitlen,
                                                                       local_traversed_hashmap_ptr,
                                                                       traversed_hash_bitlen,
                                                                       parent_indices_buffer,
                                                                       result_indices_buffer,
                                                                       1,
                                                                       result_position,
                                                                       result_buffer_size_32);
    }
    __syncthreads();

    // Check the state of the nodes in the result buffer which were not updated
    // by the compute_distance_to_child_nodes above, and if it cannot be used as
    // a parent node, it is deactivated.
    for (uint32_t i = threadIdx.x; i < result_position[0]; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index || index & index_msb_1_mask) { continue; }
      if (hashmap::search<INDEX_T, 1>(local_traversed_hashmap_ptr, traversed_hash_bitlen, index)) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering. A PASSING_TWO_HOP list holds passing nodes only: nothing to remove.
    if (!passing_two_hop) {
      for (unsigned p = threadIdx.x; p < 1; p += blockDim.x) {
        if (parent_indices_buffer[p] != invalid_index) {
          const auto parent_id =
            result_indices_buffer[parent_indices_buffer[p]] & ~index_msb_1_mask;
          if (!passes(parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_indices_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_indices_buffer[p]]   = invalid_index;
          }
        }
      }
      __syncthreads();
    }

    iter++;
  }

  // Filtering
  if (!passing_two_hop) {
    for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index) { continue; }
      index &= ~index_msb_1_mask;
      if (!passes(index)) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
    __syncthreads();
  }

  // Output search results (1st warp only).
  if (threadIdx.x < 32) {
    uint32_t offset = 0;
    for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += 32) {
      INDEX_T index = result_indices_buffer[i];
      bool is_valid = false;
      if (index != invalid_index) {
        if (index & index_msb_1_mask) {
          is_valid = true;
          index &= ~index_msb_1_mask;
        } else if ((offset < itopk_size) &&
                   hashmap::insert<INDEX_T, 1>(
                     local_traversed_hashmap_ptr, traversed_hash_bitlen, index)) {
          // If a node that is not used as a parent can be inserted into
          // the traversed hash table, it is considered a valid result.
          is_valid = true;
        }
      }
      const auto mask = __ballot_sync(0xffffffff, is_valid);
      if (is_valid) {
        const auto j = offset + __popc(mask & ((1 << threadIdx.x) - 1));
        if (j < itopk_size) {
          uint32_t k            = j + (itopk_size * (cta_id + (num_cta_per_query * row)));
          result_indices_ptr[k] = index & ~index_msb_1_mask;
          if (result_distances_ptr != nullptr) {
            DISTANCE_T dist         = result_distances_buffer[i];
            result_distances_ptr[k] = dist;
          }
        } else {
          // If it is valid and registered in the traversed hash table but is
          // not output as a result, it is removed from the hash table.
          hashmap::remove<INDEX_T>(local_traversed_hashmap_ptr, traversed_hash_bitlen, index);
        }
      }
      offset += __popc(mask);
    }
    // If the number of outputs is insufficient, fill in with invalid results.
    for (uint32_t i = offset + threadIdx.x; i < itopk_size; i += 32) {
      uint32_t k            = i + (itopk_size * (cta_id + (num_cta_per_query * row)));
      result_indices_ptr[k] = invalid_index;
      if (result_distances_ptr != nullptr) {
        result_distances_ptr[k] = utils::get_max_value<DISTANCE_T>();
      }
    }
  }

  if (threadIdx.x == 0 && cta_id == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }

#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) && (blockIdx.x == 0) &&
      ((query_id * 3) % gridDim.y < 3)) {
    printf(
      "%s:%d "
      "query, %d, thread, %d"
      ", init, %lu"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", pickup_parents, %lu"
      ", distance, %lu"
      "\n",
      __FILE__,
      __LINE__,
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_pickup_parents,
      clk_compute_distance);
  }
#endif
}

// Single-partition MULTI_CTA kernel. Thin wrapper over search_core: passes the single dataset
// descriptor/graph directly, real seeds and stats, and the standard cagra_bitset filter.
template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
__device__ void search_kernel_jit(
  IndexT* const result_indices_ptr,       // [num_queries, num_cta_per_query, itopk_size]
  DistanceT* const result_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const DataT* const queries_ptr,  // [num_queries, dataset_dim]
  const IndexT* const knn_graph,   // [dataset_size, graph_degree]
  const uint32_t max_elements,
  const uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,  // [num_queries, search_width]
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  const uint32_t visited_hash_bitlen,
  IndexT* const traversed_hashmap_ptr,  // [num_queries, 1 << traversed_hash_bitlen]
  const uint32_t traversed_hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations, /* stats */
  const IndexT graph_size,
  const uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  search_core<DataT, IndexT, DistanceT, SourceIndexT>(result_indices_ptr,
                                                      result_distances_ptr,
                                                      dataset_desc,
                                                      queries_ptr,
                                                      knn_graph,
                                                      max_elements,
                                                      graph_degree,
                                                      /*buffer_graph_degree=*/graph_degree,
                                                      source_indices_ptr,
                                                      num_distilation,
                                                      rand_xor_mask,
                                                      seed_ptr,
                                                      num_seeds,
                                                      visited_hash_bitlen,
                                                      traversed_hashmap_ptr,
                                                      traversed_hash_bitlen,
                                                      itopk_size,
                                                      min_iteration,
                                                      max_iteration,
                                                      num_executed_iterations,
                                                      graph_size,
                                                      query_id_offset,
                                                      filter_payload);
}

// Multi-partition variant of search_kernel_jit. Grid is (num_cta_per_query, num_queries,
// num_partitions); per-partition data (dataset_desc, graph, graph_degree) is read from
// partition_descs[blockIdx.z]. Cross-CTA traversed_hashmap is per-(query, partition), indexed
// by row = partition_id * num_queries + query_id (computed inside search_core). Outputs land in
// [num_partitions, num_queries, num_cta_per_query, itopk_size] partition-major. Result buffers are
// sized by max_graph_degree so the smem layout is uniform across partitions. There are no per-query
// seeds, source-index remapping, or iteration stats in multi-partition search.
template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
__device__ void search_multi_cta_mp_jit(
  const multi_partition_desc_t<DataT, IndexT, DistanceT>* partition_descs,
  IndexT* const result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const DataT* const queries_ptr,
  const uint32_t max_elements,
  const uint32_t max_graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const uint32_t visited_hash_bitlen,
  IndexT* const traversed_hashmap_ptr,
  const uint32_t traversed_hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  const uint32_t query_id_offset)
{
  const auto& part = partition_descs[blockIdx.z];

  // Each partition carries its own bitset (a view into the combined buffer); test it with the plain
  // bitset sample_filter fragment at partition-local indices.
  cagra_bitset<SourceIndexT> part_bitset{part.bitset_ptr,
                                         static_cast<SourceIndexT>(part.bitset_len),
                                         static_cast<SourceIndexT>(part.original_nbits)};
  cagra_sample_filter<SourceIndexT> filter_payload{};
  filter_payload.filter_data = (part.bitset_ptr != nullptr) ? &part_bitset : nullptr;

  search_core<DataT, IndexT, DistanceT, SourceIndexT>(result_indices_ptr,
                                                      result_distances_ptr,
                                                      part.dataset_desc,
                                                      queries_ptr,
                                                      part.graph,
                                                      max_elements,
                                                      /*graph_degree=*/part.graph_degree,
                                                      /*buffer_graph_degree=*/max_graph_degree,
                                                      /*source_indices_ptr=*/nullptr,
                                                      num_distilation,
                                                      rand_xor_mask,
                                                      /*seed_ptr=*/nullptr,
                                                      /*num_seeds=*/0,
                                                      visited_hash_bitlen,
                                                      traversed_hashmap_ptr,
                                                      traversed_hash_bitlen,
                                                      itopk_size,
                                                      min_iteration,
                                                      max_iteration,
                                                      /*num_executed_iterations=*/nullptr,
                                                      /*graph_size=*/0,
                                                      query_id_offset,
                                                      filter_payload);
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
