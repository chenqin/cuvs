/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/fp8.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cuvs::preprocessing::quantize::fp8::detail {

/** The largest finite E4M3 value: a row's scale maps its largest magnitude onto it. */
inline constexpr float max_e4m3_value = 448.0f;

/** The non-negative E4M3 value of @p code (0 to 126; 127 is NaN), increasing with the code. */
_RAFT_HOST_DEVICE inline auto e4m3_value(int code) -> float
{
  const int exponent = code >> 3;
  const int mantissa = code & 7;
  return exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9)
                       : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
}

/** The code (0 to 126) whose value is nearest to @p x >= 0, a tie going to the smaller code. */
_RAFT_HOST_DEVICE inline auto nearest_code(float x) -> uint8_t
{
  // The first code whose value is at least x, as a sorted search finds it, within [1, 126].
  int lo = 0;
  int hi = 127;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (e4m3_value(mid) < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int i       = lo < 1 ? 1 : (lo > 126 ? 126 : lo);
  const float below = e4m3_value(i - 1);
  const float above = e4m3_value(i);
  return static_cast<uint8_t>(x - below <= above - x ? i - 1 : i);
}

/** The E4M3 code of @p x: the nearest magnitude, with bit 7 set when @p x is negative. */
_RAFT_HOST_DEVICE inline auto encode(float x) -> uint8_t
{
  return static_cast<uint8_t>(nearest_code(fabsf(x)) | (x < 0.0f ? 0x80u : 0u));
}

/** One warp a row: the row's largest magnitude gives its scale, then every byte of its codes. */
template <typename T>
RAFT_KERNEL encode_rows_kernel(const T* dataset,
                               int64_t dataset_stride,
                               int64_t n_rows,
                               int64_t dim,
                               uint8_t* codes,
                               int64_t row_length,
                               float* row_scales)
{
  constexpr int warp_size = 32;
  const int64_t lane      = threadIdx.x % warp_size;
  const int64_t n_warps   = static_cast<int64_t>(gridDim.x) * (blockDim.x / warp_size);
  for (int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
       row < n_rows;
       row += n_warps) {
    const T* src = dataset + row * dataset_stride;
    float amax   = 0.0f;
    for (int64_t d = lane; d < dim; d += warp_size) {
      amax = fmaxf(amax, fabsf(static_cast<float>(src[d])));
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float scale = amax > 0.0f ? amax / max_e4m3_value : 1.0f;
    if (lane == 0) { row_scales[row] = scale; }
    uint8_t* dst = codes + row * row_length;
    for (int64_t d = lane; d < row_length; d += warp_size) {
      dst[d] = d < dim ? encode(static_cast<float>(src[d]) / scale) : uint8_t{0};
    }
  }
}

template <typename T>
void transform(raft::resources const& res,
               raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> row_scales)
{
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);
  RAFT_EXPECTS(dim <= int64_t{std::numeric_limits<uint32_t>::max()} - 15,
               "fp8::transform: the dimension (%ld) is too large",
               long(dim));
  const auto min_row_length =
    static_cast<int64_t>(get_encoded_row_length(static_cast<uint32_t>(dim)));
  RAFT_EXPECTS(codes.extent(0) == n_rows,
               "fp8::transform: codes has %ld rows but the dataset has %ld",
               long(codes.extent(0)),
               long(n_rows));
  RAFT_EXPECTS(codes.extent(1) >= min_row_length && codes.extent(1) % 16 == 0,
               "fp8::transform: the codes row length (%ld) must be a multiple of 16 and at least "
               "get_encoded_row_length(dim) = %ld",
               long(codes.extent(1)),
               long(min_row_length));
  RAFT_EXPECTS(row_scales.extent(0) == n_rows,
               "fp8::transform: row_scales has %ld entries but the dataset has %ld rows",
               long(row_scales.extent(0)),
               long(n_rows));

  // In dry-run mode the views may be unbacked or alias a shared probe buffer.
  if (raft::resource::get_dry_run_flag(res)) { return; }
  if (n_rows == 0) { return; }

  constexpr int64_t block_size     = 256;
  constexpr int64_t rows_per_block = block_size / 32;
  // Each warp loops over rows, so the grid need not cover every row at once.
  constexpr int64_t max_grid_size = 65535;
  const int64_t grid_size =
    std::min(raft::div_rounding_up_safe(n_rows, rows_per_block), max_grid_size);
  encode_rows_kernel<T><<<static_cast<uint32_t>(grid_size),
                          static_cast<uint32_t>(block_size),
                          0,
                          raft::resource::get_cuda_stream(res).get()>>>(dataset.data_handle(),
                                                                        dataset.stride(0),
                                                                        n_rows,
                                                                        dim,
                                                                        codes.data_handle(),
                                                                        codes.extent(1),
                                                                        row_scales.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::preprocessing::quantize::fp8::detail
