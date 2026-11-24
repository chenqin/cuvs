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

// PASSING_TWO_HOP version of compute_distance_to_random_nodes_jit: the initial random seed nodes
// are selected only among the nodes that pass the filter.
//
// Each of the `num_pickup` entries of the result is the nearest of up to `num_distilation` random
// nodes that pass the filter. Such a node is searched for by testing up to kMaxAttempts random
// nodes with `passes`. An entry stays invalid if no node that passes the filter is found.
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
  const IndexT seed_index_limit,
  PassT passes)
{
  constexpr IndexT invalid_index  = ~static_cast<IndexT>(0);
  constexpr uint32_t kMaxAttempts = 32;  // random nodes tested to find one that passes
  // Multiplied by the attempt number and added to gid, so that every attempt uses a different
  // random number (2^64 / golden ratio).
  constexpr uint64_t kTryIncrement = 0x9E3779B97F4A7C15ull;

  const uint32_t team_size_bits = smem_desc->team_size_bitshift_from_smem();
  const auto args_load          = smem_desc->args.load();
  const auto max_i = raft::round_up_safe<uint32_t>(num_pickup, device::warp_size >> team_size_bits);

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
        for (uint32_t t = 0; t < kMaxAttempts && seed_index == invalid_index; t++) {
          const uint64_t r = cuvs::neighbors::detail::device::xorshift64(
            (gid + kTryIncrement * (t + 1)) ^ rand_xor_mask);
          const IndexT node = static_cast<IndexT>((r >> 8) % seed_index_limit);
          if (passes(node)) { seed_index = node; }
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
          // Deactivate this entry as insertion into visited hash table has failed.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        } else if (hashmap::search<IndexT, 1>(
                     traversed_hash_ptr, traversed_hash_bitlen, best_index_team_local)) {
          // Deactivate this entry as it has been already used by others.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        }
      }
      result_distances_ptr[i] = best_norm2_team_local;
      result_indices_ptr[i]   = best_index_team_local;
    }
  }
}

// PASSING_TWO_HOP version of compute_distance_to_child_nodes_jit, for one parent node. The child
// nodes of the parent are
//   - first hop: the neighbors of the parent that pass the filter, and
//   - second hop: for every neighbor of the parent that is filtered out (a "bridge"), the
//     neighbors of the bridge that pass the filter.
// Distances are computed for the child nodes only; a filtered-out node is never written to the
// result buffer.
//
// The child nodes are written to `result_indices_ptr` / `result_distances_ptr` from position
// `max_result_position - 1` downwards, and never below `min_result_position`. The first hop is
// completed before the second hop starts, and child nodes that do not fit are dropped.
// `result_position[0]` must be `max_result_position` on entry. On return, it is the lowest
// position that holds a child node (`max_result_position` if there is none).
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
  PassT passes)
{
  constexpr IndexT invalid_index = ~static_cast<IndexT>(0);

  // Add `child` to the child nodes unless it cannot be inserted into the visited hash table of
  // this CTA (it is already registered, or the table is full) or a CTA of the same query has
  // already used it as a parent. A child node that does not fit in the result buffer is dropped.
  auto take = [&](const IndexT child) {
    if (hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child) == 0) { return; }
    if (hashmap::search<IndexT, 1>(traversed_hashmap_ptr, traversed_hash_bitlen, child)) { return; }
    const int j = atomicSub(result_position, 1) - 1;
    if (j >= min_result_position) { result_indices_ptr[j] = child; }
  };

  // First hop: a neighbor of the parent that passes the filter becomes a child node. A neighbor
  // that is filtered out is stored in `bridges`; its neighbors are examined in the second hop.
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

  // Second hop: examine the neighbors of every bridge until the child part of the result buffer
  // is full.
  //
  // Slot `s` is entry `s % knn_k` of the graph row of `bridges[s / knn_k]`. In one round (one
  // iteration of the loop below) a thread handles `unroll` slots. It first loads all their graph
  // entries and only then tests them, so that the loads are independent of each other. `unroll` is
  // kSlotsPerRound / blockDim.x, limited to the range [1, kMaxUnroll]: the thread block handles
  // kSlotsPerRound slots per round, or one slot per thread when it has more than kSlotsPerRound
  // threads.
  //
  // A thread checks whether the buffer is full once per round, and reads `result_position[0]`
  // without synchronization for that. This is safe: the value only decreases, so a stale value
  // only delays the exit, and take() drops the child nodes that do not fit. The child nodes that
  // are registered in the visited hash table in the meantime are the reason for the extra entries
  // that search_plan.cuh adds to it (kTwoHopExtraVisitedNodes). When the table is full, take()
  // drops the child node.
  constexpr uint32_t kSlotsPerRound = 256;  // slots per round for the thread block
  constexpr uint32_t kMaxUnroll     = 4;    // max slots per round for a thread
  const uint32_t num_slots          = knn_k * knn_k;
  const uint32_t slots_per_thread   = kSlotsPerRound / blockDim.x;
  const uint32_t unroll =
    slots_per_thread < 1 ? 1u : (slots_per_thread > kMaxUnroll ? kMaxUnroll : slots_per_thread);
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
    for (uint32_t u = 0; u < kMaxUnroll; u++) {
      if (child[u] != invalid_index && passes(child[u])) { take(child[u]); }
    }
  }
  __syncthreads();

  // Compute the distances of the child nodes, as compute_distance_to_child_nodes_jit does.
  const auto team_size_bits = smem_desc->team_size_bitshift_from_smem();
  const int first =
    result_position[0] < min_result_position ? min_result_position : result_position[0];
  const auto max_i =
    raft::round_up_safe<uint32_t>(static_cast<uint32_t>(max_result_position - min_result_position),
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
  // take() decrements result_position[0] even when a child node does not fit, so it can be below
  // min_result_position. Set it to the lowest position that holds a child node, which is what
  // the caller expects.
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

  // PASSING_TWO_HOP is used only without seed nodes (seed_ptr == nullptr). Seed nodes supplied by
  // the caller are not filtered, so a search with seed nodes uses the SCORE_ALL traversal, which
  // removes the filtered-out nodes from the result buffer. The host does not pass seed nodes to
  // this kernel at present.
  const bool passing_two_hop =
    (filter_payload.flags & kCagraFilterPassingTwoHop) != 0 && seed_ptr == nullptr;

  // Whether a node passes the filter: the extern sample_filter function (linked via JIT LTO).
  auto passes = [&](const INDEX_T node) -> bool {
    return sample_filter<SourceIndexT>(
      query_id + query_id_offset, to_source_index(node), filter_payload.sample_filter_data());
  };

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
  // [graph_degree] Neighbors of the parent that are filtered out (bridges). The host adds this
  // buffer to the shared memory only for PASSING_TWO_HOP (see smem_size in search_multi_cta.cuh).
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

  if (passing_two_hop) {
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

    // Filtering - remove the parent from the result buffer if it is filtered out. Not needed for
    // PASSING_TWO_HOP, whose result buffer only holds nodes that pass the filter.
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

  // Filtering - remove the filtered-out nodes from the result buffer. Not needed for
  // PASSING_TWO_HOP (see above).
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
