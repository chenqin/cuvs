/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/fp8.h>
#include <cuvs/preprocessing/quantize/fp8.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace {

template <typename T>
void _transform(cuvsResources_t res,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* codes_tensor,
                DLManagedTensor* row_scales_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  using dataset_mdspan_type    = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using codes_mdspan_type      = raft::device_matrix_view<uint8_t, int64_t, raft::row_major>;
  using row_scales_mdspan_type = raft::device_vector_view<float, int64_t>;

  cuvs::preprocessing::quantize::fp8::transform(
    *res_ptr,
    cuvs::core::from_dlpack<dataset_mdspan_type>(dataset_tensor),
    cuvs::core::from_dlpack<codes_mdspan_type>(codes_tensor),
    cuvs::core::from_dlpack<row_scales_mdspan_type>(row_scales_tensor));
}

}  // namespace

extern "C" cuvsError_t cuvsFp8QuantizerEncodedRowLength(uint32_t dim, uint32_t* encoded_row_length)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(encoded_row_length != nullptr, "cuvsFp8QuantizerEncodedRowLength: null output");
    *encoded_row_length = cuvs::preprocessing::quantize::fp8::get_encoded_row_length(dim);
  });
}

extern "C" cuvsError_t cuvsFp8QuantizerTransform(cuvsResources_t res,
                                                 DLManagedTensor* dataset_tensor,
                                                 DLManagedTensor* codes_tensor,
                                                 DLManagedTensor* row_scales_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(
      dataset_tensor != nullptr && codes_tensor != nullptr && row_scales_tensor != nullptr,
      "cuvsFp8QuantizerTransform: null tensor");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(dataset_tensor->dl_tensor) &&
                   cuvs::core::is_dlpack_device_compatible(codes_tensor->dl_tensor) &&
                   cuvs::core::is_dlpack_device_compatible(row_scales_tensor->dl_tensor),
                 "cuvsFp8QuantizerTransform: dataset, codes and row_scales must be in device "
                 "memory");
    auto dtype = dataset_tensor->dl_tensor.dtype;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      _transform<float>(res, dataset_tensor, codes_tensor, row_scales_tensor);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      _transform<half>(res, dataset_tensor, codes_tensor, row_scales_tensor);
    } else {
      RAFT_FAIL("cuvsFp8QuantizerTransform: unsupported dataset dtype: code=%d, bits=%d",
                dtype.code,
                dtype.bits);
    }
  });
}
