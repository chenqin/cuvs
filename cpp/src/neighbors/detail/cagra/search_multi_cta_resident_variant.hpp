/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "jit_lto_kernels/kernel_def.hpp"  // kSmallResidentMaxRegisters
#include "search_multi_cta_waves.hpp"

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <rtcx/algorithm_launcher.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

// The `search_multi_cta_small` entry point has a resident variant, linked with a lower register
// cap (kSmallResidentMaxRegisters, kernel_def.hpp) so that a multiprocessor holds more
// of its thread blocks. select_and_run() launches it instead of the default variant only when it
// runs the grid in fewer waves of thread blocks; otherwise the default variant's higher register
// cap, which spills less, is faster. On GPUs whose thread or shared memory limits already bind at
// the default cap (multiprocessors of 1,536 threads, compute capability 8.6, 8.9 and 12.0) the
// resident variant cannot hold more blocks, and it is never linked.
//
// The caches below are per thread. They assume that a thread keeps the CUDA context it uses for a
// device, so that the number of multiprocessors it sees (which a CUDA MPS client can limit) stays
// the same.

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

/**
 * Dynamic shared memory per thread block above which the launcher keeps the default variant.
 * Beyond 48 KiB at most four blocks fit on a multiprocessor, which the default variant's register
 * cap already allows for every block size of the small entry point.
 */
inline constexpr std::uint32_t kResidentVariantMaxSmem = 48 * 1024;

/**
 * Whether the launcher may select the resident variant: not when the environment sets a register
 * cap for the multi-CTA search kernels (-maxrregcount in CUVS_CAGRA_JIT_LINK_OPTIONS,
 * CUVS_CAGRA_JIT_LINK_OPTIONS_search_multi_cta or
 * CUVS_CAGRA_JIT_LINK_OPTIONS_search_multi_cta_small), which then applies to every launch, and not
 * when CUVS_CAGRA_MULTI_CTA_RESIDENT_VARIANT is set to 0. Read once per process.
 */
inline bool resident_variant_enabled()
{
  static const bool enabled = [] {
    const char* setting = std::getenv("CUVS_CAGRA_MULTI_CTA_RESIDENT_VARIANT");
    if (setting != nullptr && std::strcmp(setting, "0") == 0) { return false; }
    for (const char* name : {"CUVS_CAGRA_JIT_LINK_OPTIONS",
                             "CUVS_CAGRA_JIT_LINK_OPTIONS_search_multi_cta",
                             "CUVS_CAGRA_JIT_LINK_OPTIONS_search_multi_cta_small"}) {
      const char* options = std::getenv(name);
      if (options != nullptr && std::strstr(options, "-maxrregcount") != nullptr) { return false; }
    }
    return true;
  }();
  return enabled;
}

/** The limits of a device's multiprocessors that do not depend on a kernel's registers. */
struct multiprocessor_limits {
  std::uint32_t num_sms             = 0;
  std::uint32_t max_threads         = 0;
  std::uint32_t max_blocks          = 0;
  std::uint32_t max_smem            = 0;
  std::uint32_t reserved_smem_block = 0;
  /// Shared memory allocation unit: 256 bytes on compute capability 7.x, 128 from 8.0.
  std::uint32_t smem_unit = 128;
};

/** The multiprocessor limits of `device`, cached per thread. */
inline const multiprocessor_limits& device_multiprocessor_limits(int device)
{
  thread_local std::unordered_map<int, multiprocessor_limits> cache;
  if (auto found = cache.find(device); found != cache.end()) { return found->second; }
  auto attribute = [device](cudaDeviceAttr attr) {
    int value = 0;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&value, attr, device));
    return static_cast<std::uint32_t>(value > 0 ? value : 0);
  };
  multiprocessor_limits limits;
  limits.num_sms             = attribute(cudaDevAttrMultiProcessorCount);
  limits.max_threads         = attribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  limits.max_blocks          = attribute(cudaDevAttrMaxBlocksPerMultiprocessor);
  limits.max_smem            = attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor);
  limits.reserved_smem_block = attribute(cudaDevAttrReservedSharedMemoryPerBlock);
  limits.smem_unit           = attribute(cudaDevAttrComputeCapabilityMajor) < 8 ? 256 : 128;
  return cache.emplace(device, limits).first->second;
}

/** The registers per thread and static shared memory of a kernel, as cudaFuncGetAttributes reads
 * them. */
struct kernel_resources {
  std::uint32_t registers   = 0;
  std::uint32_t static_smem = 0;
  bool known                = false;
};

/**
 * The registers and static shared memory of `kernel` (not known when cudaFuncGetAttributes fails
 * or another CUDA error is pending). Cached per thread for each kernel.
 */
inline kernel_resources query_kernel_resources(cudaKernel_t kernel)
{
  thread_local std::unordered_map<const void*, kernel_resources> cache;
  const void* key = static_cast<const void*>(kernel);
  if (auto found = cache.find(key); found != cache.end()) { return found->second; }
  if (cudaPeekAtLastError() != cudaSuccess) { return {}; }
  kernel_resources resources;
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(&attributes, kernel) == cudaSuccess) {
    resources.registers   = static_cast<std::uint32_t>(attributes.numRegs);
    resources.static_smem = static_cast<std::uint32_t>(attributes.sharedSizeBytes);
    resources.known       = true;
  } else {
    (void)cudaGetLastError();
  }
  cache.emplace(key, resources);
  return resources;
}

/**
 * The most thread blocks of `block_size` threads with `smem_size` bytes of dynamic and
 * `static_smem` bytes of static shared memory that a multiprocessor holds by its thread, block and
 * shared memory limits, whatever the registers of the kernel.
 */
inline std::uint32_t blocks_by_multiprocessor_limits(const multiprocessor_limits& limits,
                                                     std::uint32_t block_size,
                                                     std::uint32_t smem_size,
                                                     std::uint32_t static_smem)
{
  if (block_size == 0) { return 0; }
  std::uint64_t blocks = limits.max_threads / block_size;
  if (limits.max_blocks > 0) { blocks = std::min<std::uint64_t>(blocks, limits.max_blocks); }
  const std::uint64_t unit = limits.smem_unit;
  const std::uint64_t smem_per_block =
    (static_cast<std::uint64_t>(smem_size) + static_smem + unit - 1) / unit * unit +
    limits.reserved_smem_block;
  if (smem_per_block > 0) {
    blocks = std::min<std::uint64_t>(blocks, limits.max_smem / smem_per_block);
  }
  return static_cast<std::uint32_t>(blocks);
}

/**
 * Thread blocks of `kernel` that a multiprocessor of `device` holds at a time: what the CUDA
 * occupancy calculator reports without dynamic shared memory (the register, thread and block
 * limits; this does not depend on the kernel's dynamic shared memory attribute, which only the
 * launch sets), capped by the shared memory limit. 0, so that the caller keeps the default
 * variant, when the calculator fails or another CUDA error is pending. Cached per thread for each
 * kernel, device, block size and shared memory size.
 */
inline std::uint32_t kernel_blocks_per_sm(cudaKernel_t kernel,
                                          int device,
                                          const multiprocessor_limits& limits,
                                          std::uint32_t block_size,
                                          std::uint32_t smem_size)
{
  struct key_hash {
    std::size_t operator()(const std::pair<const void*, std::uint64_t>& key) const
    {
      return std::hash<const void*>{}(key.first) ^ (std::hash<std::uint64_t>{}(key.second) << 1);
    }
  };
  thread_local std::unordered_map<std::pair<const void*, std::uint64_t>, std::uint32_t, key_hash>
    cache;
  // device < 256, block_size <= 1024 and smem_size < 2^32 occupy separate bits.
  const std::uint64_t shape = (static_cast<std::uint64_t>(device) << 56) |
                              (static_cast<std::uint64_t>(block_size) << 32) | smem_size;
  const std::pair<const void*, std::uint64_t> key{static_cast<const void*>(kernel), shape};
  if (auto found = cache.find(key); found != cache.end()) { return found->second; }

  // Do not query, and do not clear, while an error of other work is pending.
  if (cudaPeekAtLastError() != cudaSuccess) { return 0; }
  int occupancy = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, static_cast<int>(block_size), 0) != cudaSuccess) {
    (void)cudaGetLastError();
    occupancy = 0;
  }
  const kernel_resources resources = query_kernel_resources(kernel);
  const std::uint32_t blocks =
    std::min(static_cast<std::uint32_t>(occupancy > 0 ? occupancy : 0),
             blocks_by_multiprocessor_limits(limits, block_size, smem_size, resources.static_smem));
  cache.emplace(key, blocks);
  return blocks;
}

/**
 * The resident variant's launcher for a launch of `grid_blocks` thread blocks of `block_size`
 * threads with `smem_size` bytes of dynamic shared memory, or nullptr to launch
 * `default_launcher`: the resident variant if prefer_resident_variant() holds for the two
 * variants' blocks per multiprocessor on the current device.
 *
 * `make_resident` returns the resident variant's launcher, linking it the first time. It is called
 * on the first launch, on each thread, that needs more than one wave of the default variant and
 * whose grid would need fewer at the multiprocessor limits, so processes whose grids fit in one
 * wave never link it (the link adds tens of milliseconds to that launch). If it fails, the default
 * variant is used from then on.
 */
template <typename MakeResident>
std::shared_ptr<rtcx::algorithm_launcher> select_resident_launcher(
  const std::shared_ptr<rtcx::algorithm_launcher>& default_launcher,
  MakeResident&& make_resident,
  std::uint64_t grid_blocks,
  std::uint32_t block_size,
  std::uint32_t smem_size)
{
  if (smem_size > kResidentVariantMaxSmem || block_size == 0) { return nullptr; }
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  const auto& limits = device_multiprocessor_limits(device);
  const std::uint32_t default_blocks =
    kernel_blocks_per_sm(default_launcher->get_kernel(), device, limits, block_size, smem_size);
  if (default_blocks == 0 ||
      waves_of_thread_blocks(grid_blocks, limits.num_sms, default_blocks) <= 1) {
    return nullptr;
  }
  // Do not link the resident variant when it cannot save a wave: the default kernel already uses
  // no more registers than the resident cap, or no register count could save one.
  const kernel_resources resources = query_kernel_resources(default_launcher->get_kernel());
  if (!resources.known || resources.registers <= kSmallResidentMaxRegisters ||
      !prefer_resident_variant(
        grid_blocks,
        limits.num_sms,
        default_blocks,
        blocks_by_multiprocessor_limits(limits, block_size, smem_size, resources.static_smem))) {
    return nullptr;
  }

  thread_local std::unordered_map<const void*, std::shared_ptr<rtcx::algorithm_launcher>>
    resident_of;
  const void* key = static_cast<const void*>(default_launcher->get_kernel());
  auto found      = resident_of.find(key);
  if (found == resident_of.end()) {
    std::shared_ptr<rtcx::algorithm_launcher> resident_launcher;
    try {
      resident_launcher = make_resident();
    } catch (const std::exception& e) {
      RAFT_LOG_WARN(
        "CAGRA multi-CTA search: the resident variant could not be linked (%s); "
        "using the default variant",
        e.what());
    }
    found = resident_of.emplace(key, std::move(resident_launcher)).first;
  }
  if (!found->second) { return nullptr; }
  const std::uint32_t resident_blocks =
    kernel_blocks_per_sm(found->second->get_kernel(), device, limits, block_size, smem_size);
  return prefer_resident_variant(grid_blocks, limits.num_sms, default_blocks, resident_blocks)
           ? found->second
           : nullptr;
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
