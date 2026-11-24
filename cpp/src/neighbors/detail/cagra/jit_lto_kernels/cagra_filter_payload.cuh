/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

// Flags stored in cagra_sample_filter::flags. Only the single-partition MULTI_CTA launcher sets
// them (see set_multi_cta_filter_flags). For every other kernel `flags` is 0 and the filter is
// evaluated by the linked sample_filter.

/// `search_params::filter_expansion` is PASSING_TWO_HOP.
inline constexpr std::uint32_t kCagraFilterPassingTwoHop = 1u << 0;
/// `filter_data` points to a `bitset_filter_data_t` with standard 32-bit packing: row `i` passes
/// the filter when bit `i % 32` of `bitset_ptr[i / 32]` is set. The kernel tests the words
/// directly instead of calling the linked `sample_filter`. Set only together with
/// `kCagraFilterPassingTwoHop`.
inline constexpr std::uint32_t kCagraFilterBitsetWords = 1u << 1;
/// `filter_data` is `search_params::query_bitsets`: an array of `const std::uint32_t*` indexed by
/// `query_id + query_id_offset`. Each pointer is the bitset of one query (standard 32-bit
/// packing), or null if the query is not filtered. The linked `sample_filter` is not called.
inline constexpr std::uint32_t kCagraFilterQueryBitsets = 1u << 2;

/// Device payload for linked CAGRA sample filters plus query offset for wrapped filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  void* filter_data{nullptr};
  std::uint32_t query_id_offset{0};
  std::uint32_t flags{0};  // Bitwise OR of kCagraFilter* flags

  __device__ __forceinline__ void* sample_filter_data() { return filter_data; }
};

}  // namespace cuvs::neighbors::cagra::detail
