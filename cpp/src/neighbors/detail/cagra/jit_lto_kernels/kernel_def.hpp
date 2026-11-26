/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuvs/neighbors/common.hpp>

#include "../compute_distance.hpp"  // dataset_descriptor_base_t
#include "../multi_partition_desc.hpp"
#include "cagra_bitset.cuh"          // multi-partition payload helpers
#include "cagra_filter_payload.cuh"  // single-partition / UDF payload
#include "search_single_cta_device_helpers.cuh"

namespace cuvs::neighbors::cagra::detail {

// Function types for extern "C" __global__ JIT entry points — must match cudaLibraryGetKernel /
// rtcx::algorithm_launcher::dispatch signatures exactly (see static_assert in each *_kernel.cu).

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_single_cta_kernel_func_t =
  void(uintptr_t,
       DistanceT* const,
       const std::uint32_t,
       const DataT* const,
       const IndexT* const,
       const std::uint32_t,
       const SourceIndexT*,
       const unsigned,
       const uint64_t,
       const IndexT*,
       const uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       std::uint32_t* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const IndexT,
       cagra_sample_filter<SourceIndexT>);

namespace single_cta_search {

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_single_cta_mp_kernel_func_t =
  void(const multi_partition_desc_t<DataT, IndexT, DistanceT>*,
       const DataT* const,
       IndexT* const,
       DistanceT* const,
       const std::uint32_t,
       const unsigned,
       const uint64_t,
       const std::uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       std::uint32_t* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t);

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_single_cta_p_kernel_func_t =
  void(worker_handle_t*,
       job_desc_t<job_desc_traits<DataT, IndexT, DistanceT>>*,
       uint32_t*,
       const IndexT* const,
       const std::uint32_t,
       const SourceIndexT*,
       const unsigned,
       const uint64_t,
       const IndexT*,
       const uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       std::uint32_t* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const IndexT,
       cagra_sample_filter<SourceIndexT>);

}  // namespace single_cta_search

namespace multi_cta_search {

/// Maximum thread block size of the `search_multi_cta_small` entry point: the block size its
/// default register cap is chosen for, and the largest thread block size for which the launcher
/// selects it.
inline constexpr std::uint32_t kSmallBlockMaxThreads = 256;

/// Minimum number of resident thread blocks per multiprocessor for `search_multi_cta_small`. Its
/// planner limits the registers per thread at link time (-maxrregcount) so that this many thread
/// blocks of `kSmallBlockMaxThreads` threads fit on a multiprocessor. On GPUs whose
/// multiprocessors hold fewer threads than that (compute capability 7.5) the cap would cost spills
/// without adding resident blocks, and the launcher does not select `search_multi_cta_small`.
inline constexpr std::uint32_t kSmallBlockMinBlocksPerSm = 6;

/// Register cap of the resident variant of `search_multi_cta_small`: the most registers a thread
/// can have when 2,048 threads, the most a multiprocessor of compute capability 8.0, 9.0 or 10.0
/// holds, share its 65,536 registers (65,536 / 2,048 = 32). Such a multiprocessor then holds 32
/// blocks of 64 threads, against 24 at the default cap of 40. The launcher selects the variant per
/// launch (see search_multi_cta_resident_variant.hpp).
inline constexpr std::uint32_t kSmallResidentMaxRegisters = 65536u / 2048u;
static_assert(kSmallResidentMaxRegisters == 32);

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_multi_cta_mp_kernel_func_t =
  void(const multi_partition_desc_t<DataT, IndexT, DistanceT>*,
       IndexT* const,
       DistanceT* const,
       const DataT* const,
       const std::uint32_t,
       const std::uint32_t,
       const unsigned,
       const uint64_t,
       const std::uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t);

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_multi_cta_kernel_func_t =
  void(IndexT* const,
       DistanceT* const,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const DataT* const,
       const IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const SourceIndexT*,
       const unsigned,
       const uint64_t,
       const IndexT*,
       const std::uint32_t,
       const std::uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       std::uint32_t* const,
       const IndexT,
       const std::uint32_t,
       cagra_sample_filter<SourceIndexT>);

}  // namespace multi_cta_search

namespace multi_kernel_search {

template <typename DataT, typename IndexT, typename DistanceT>
using random_pickup_kernel_func_t = void(const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
                                         const DataT* const,
                                         const std::size_t,
                                         const unsigned,
                                         const uint64_t,
                                         const IndexT*,
                                         const std::uint32_t,
                                         IndexT* const,
                                         DistanceT* const,
                                         const std::uint32_t,
                                         IndexT* const,
                                         const std::uint32_t,
                                         const IndexT);

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using compute_distance_to_child_nodes_kernel_func_t =
  void(const IndexT* const,
       IndexT* const,
       DistanceT* const,
       const std::size_t,
       const std::uint32_t,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const IndexT* const,
       const std::uint32_t,
       const SourceIndexT*,
       const DataT*,
       IndexT* const,
       const std::uint32_t,
       IndexT* const,
       DistanceT* const,
       const std::uint32_t,
       cagra_sample_filter<SourceIndexT>);

template <typename IndexT, typename DistanceT, typename SourceIndexT>
using apply_filter_kernel_func_t = void(const SourceIndexT* const,
                                        IndexT* const,
                                        DistanceT* const,
                                        const std::size_t,
                                        const std::uint32_t,
                                        const std::uint32_t,
                                        const std::uint32_t,
                                        cagra_sample_filter<SourceIndexT>);

}  // namespace multi_kernel_search

}  // namespace cuvs::neighbors::cagra::detail
