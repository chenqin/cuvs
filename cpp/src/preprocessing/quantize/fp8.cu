/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/fp8.cuh"

#include <cuvs/preprocessing/quantize/fp8.hpp>

#include <raft/core/device_mdarray.hpp>

namespace cuvs::preprocessing::quantize::fp8 {

#define CUVS_INST_QUANTIZATION(T)                                                     \
  void transform(raft::resources const& res,                                          \
                 raft::device_matrix_view<const T, int64_t, raft::row_major> dataset, \
                 raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,   \
                 raft::device_vector_view<float, int64_t> row_scales)                 \
  {                                                                                   \
    detail::transform<T>(res, dataset, codes, row_scales);                            \
  }                                                                                   \
  void transform(raft::resources const& res,                                          \
                 raft::device_matrix_view<const T, int64_t, raft::row_major> dataset, \
                 quantizer& out)                                                      \
  {                                                                                   \
    detail::transform<T>(res, dataset, out.codes.view(), out.row_scales.view());      \
  }

CUVS_INST_QUANTIZATION(float);
CUVS_INST_QUANTIZATION(half);

#undef CUVS_INST_QUANTIZATION

quantizer::quantizer(raft::resources const& res, int64_t n_rows, uint32_t dim)
  : codes{raft::make_device_matrix<uint8_t, int64_t>(
      res, n_rows, static_cast<int64_t>(get_encoded_row_length(dim)))},
    row_scales{raft::make_device_vector<float, int64_t>(res, n_rows)}
{
}

}  // namespace cuvs::preprocessing::quantize::fp8
