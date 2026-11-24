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

/// Device payload for linked CAGRA sample filters plus query offset for wrapped filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  void* filter_data{nullptr};
  std::uint32_t query_id_offset{0};
  std::uint32_t flags{0};  // Bitwise OR of kCagraFilter* flags

  __device__ __forceinline__ void* sample_filter_data() { return filter_data; }
};

}  // namespace cuvs::neighbors::cagra::detail
