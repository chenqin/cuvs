/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

/**
 * Waves of thread blocks in a grid of `grid_blocks` blocks when `num_sms` multiprocessors each hold
 * `blocks_per_sm` of them at a time (0 when the grid cannot run).
 */
constexpr std::uint64_t waves_of_thread_blocks(std::uint64_t grid_blocks,
                                               std::uint32_t num_sms,
                                               std::uint32_t blocks_per_sm)
{
  const std::uint64_t per_wave = static_cast<std::uint64_t>(num_sms) * blocks_per_sm;
  return per_wave == 0 ? 0 : (grid_blocks + per_wave - 1) / per_wave;
}

/**
 * Whether a launch of `grid_blocks` thread blocks on `num_sms` multiprocessors should use the
 * resident variant of the small multi-CTA entry point, which holds `resident_blocks_per_sm` blocks
 * on a multiprocessor, over the default variant, which holds `default_blocks_per_sm`: only if the
 * resident variant runs the grid in fewer waves. When both need the same number of waves the
 * default variant is kept, because its higher register cap spills less (see
 * search_multi_cta_resident_variant.hpp).
 */
constexpr bool prefer_resident_variant(std::uint64_t grid_blocks,
                                       std::uint32_t num_sms,
                                       std::uint32_t default_blocks_per_sm,
                                       std::uint32_t resident_blocks_per_sm)
{
  if (default_blocks_per_sm == 0 || resident_blocks_per_sm <= default_blocks_per_sm) {
    return false;
  }
  return waves_of_thread_blocks(grid_blocks, num_sms, resident_blocks_per_sm) <
         waves_of_thread_blocks(grid_blocks, num_sms, default_blocks_per_sm);
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
