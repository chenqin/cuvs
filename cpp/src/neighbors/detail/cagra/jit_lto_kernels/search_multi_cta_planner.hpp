/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag>
struct CagraMultiCtaSearchPlanner
  : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag> {
  static inline rtcx::launcher_jit_cache launcher_jit_cache{};
  static inline rtcx::launcher_jit_cache launcher_jit_cache_small{};

  /// `small` selects the `search_multi_cta_small` entry point, which is compiled for thread blocks
  /// of at most `kSmallBlockMaxThreads` threads. Each entry point has its own launcher cache: both
  /// are linked from the same fragments, and a cache reuses the linked kernel for a fragment key.
  CagraMultiCtaSearchPlanner(cuvs::distance::DistanceType /*metric*/,
                             uint32_t /*team_size*/,
                             uint32_t /*dataset_block_dim*/,
                             bool /*is_vpq*/,
                             uint32_t /*pq_bits*/,
                             uint32_t /*pq_len*/,
                             bool small = false)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag>(
        small ? "search_multi_cta_small" : "search_multi_cta",
        small ? launcher_jit_cache_small : launcher_jit_cache)
  {
    // The kernel has no __launch_bounds__ (see search_multi_cta_kernel.cu.in): cap its registers at
    // 64, what 1,024 threads a block allow, unless the environment set a cap.
    bool capped = false;
    for (const auto& option : this->linktime_extra_options) {
      if (option.rfind("-maxrregcount", 0) == 0) { capped = true; }
    }
    if (!capped) { this->linktime_extra_options.push_back("-maxrregcount=64"); }
  }

  void add_search_multi_cta_kernel_fragment()
  {
    this->template add_static_fragment<
      fragment_tag_search_multi_cta<DataTag, SourceIndexTag, IndexTag, DistanceTag>>();
  }
};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag>
struct CagraMultiCtaMpSearchPlanner
  : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag> {
  static inline rtcx::launcher_jit_cache launcher_jit_cache{};

  CagraMultiCtaMpSearchPlanner(cuvs::distance::DistanceType /*metric*/,
                               uint32_t /*team_size*/,
                               uint32_t /*dataset_block_dim*/,
                               bool /*is_vpq*/,
                               uint32_t /*pq_bits*/,
                               uint32_t /*pq_len*/)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag>(
        "search_multi_cta_mp", launcher_jit_cache)
  {
  }

  void add_search_multi_cta_kernel_fragment()
  {
    this->template add_static_fragment<
      fragment_tag_search_multi_cta_mp<DataTag, SourceIndexTag, IndexTag, DistanceTag>>();
  }
};

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
