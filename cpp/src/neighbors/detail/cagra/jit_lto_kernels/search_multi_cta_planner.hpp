/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include "kernel_def.hpp"
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/distance/distance.hpp>

#include <algorithm>
#include <cstdint>
#include <string>

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
  static inline rtcx::launcher_jit_cache launcher_jit_cache_small_resident{};

  /// `small` selects the `search_multi_cta_small` entry point, which is used only for thread blocks
  /// of at most `kSmallBlockMaxThreads` threads; with `resident` as well, its resident variant,
  /// linked with the lower register cap kSmallResidentMaxRegisters (see
  /// search_multi_cta_resident_variant.hpp). Each entry point and variant has its own launcher
  /// cache: they are linked from the same fragments, and a cache reuses the linked kernel for a
  /// fragment key.
  CagraMultiCtaSearchPlanner(cuvs::distance::DistanceType /*metric*/,
                             uint32_t /*team_size*/,
                             uint32_t /*dataset_block_dim*/,
                             bool /*is_vpq*/,
                             uint32_t /*pq_bits*/,
                             uint32_t /*pq_len*/,
                             bool small    = false,
                             bool resident = false)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag>(
        small ? "search_multi_cta_small" : "search_multi_cta",
        small ? (resident ? launcher_jit_cache_small_resident : launcher_jit_cache_small)
              : launcher_jit_cache)
  {
    auto& options = this->linktime_extra_options;
    if (small) {
      // search_multi_cta_small is search_multi_cta for smaller blocks: the options given for
      // search_multi_cta apply to it as well, before the general ones and its own.
      auto inherited = this->link_options_from_env("CUVS_CAGRA_JIT_LINK_OPTIONS_search_multi_cta");
      options.insert(options.begin(), inherited.begin(), inherited.end());
    }
    // Neither entry point has __launch_bounds__ (see search_multi_cta_kernel.cu.in), so the link
    // sets the register cap. The last -maxrregcount given wins. Without one, the cap is what launch
    // bounds would allow: 64 registers for blocks of up to 1,024 threads, and kSmallMaxRegisters
    // for the small entry point (kSmallResidentMaxRegisters for its resident variant, which the
    // launcher does not select when the environment gives a cap).
    auto is_cap = [](const std::string& option) { return option.rfind("-maxrregcount", 0) == 0; };
    auto last   = std::find_if(options.rbegin(), options.rend(), is_cap);
    if (last == options.rend()) {
      uint32_t cap = 64;
      if (small) { cap = resident ? kSmallResidentMaxRegisters : kSmallMaxRegisters; }
      options.push_back("-maxrregcount=" + std::to_string(cap));
    } else {
      std::string cap = *last;
      options.erase(std::remove_if(options.begin(), options.end(), is_cap), options.end());
      options.push_back(cap);
    }
  }

  /// The register cap of search_multi_cta_small without a -maxrregcount option: the most registers
  /// a thread can have when kSmallBlockMinBlocksPerSm blocks of kSmallBlockMaxThreads threads share
  /// a multiprocessor's 65,536 registers, rounded down to the 8-register allocation unit
  /// (65,536 / (6 x 256) = 42, so 40).
  static constexpr uint32_t kSmallMaxRegisters =
    (65536u / (kSmallBlockMinBlocksPerSm * kSmallBlockMaxThreads)) / 8u * 8u;
  static_assert(kSmallMaxRegisters == 40);

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
