/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

/// Bits of cagra_sample_filter::flags. They are set on the host for the MULTI_CTA kernel only;
/// every other kernel leaves flags at 0 and calls the linked sample_filter.
enum cagra_sample_filter_flags : std::uint32_t {
  /// filter_data points at a bitset_filter_data_t whose words are tested as
  /// `(word[i / 32] >> (i % 32)) & 1`: the kernel reads the words itself, one load a test,
  /// instead of calling the linked sample_filter for every node.
  kCagraFilterBitsetWords = 1u << 0,
  /// filter_data points at one `const std::uint32_t*` per query (indexed by query id plus
  /// query_id_offset), each the words of that query's bitset. The linked sample_filter is never
  /// called.
  kCagraFilterQueryBitsets = 1u << 1,
  /// search_params::filter_expansion == PASSING_TWO_HOP.
  kCagraFilterPassingTwoHop = 1u << 2,
};

/// Device payload for linked CAGRA sample filters plus query offset for wrapped filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  void* filter_data{nullptr};
  std::uint32_t query_id_offset{0};
  std::uint32_t flags{0};  // cagra_sample_filter_flags; occupies what was padding

  __device__ __forceinline__ void* sample_filter_data() { return filter_data; }
};

}  // namespace cuvs::neighbors::cagra::detail
