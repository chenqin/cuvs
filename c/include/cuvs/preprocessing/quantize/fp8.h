/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup preprocessing_c_fp8 C API for FP8 (E4M3) Quantizer
 * @{
 */

/**
 * @brief Bytes of one row of codes: the dimension rounded up to a multiple of 16.
 *
 * @param[in] dim number of components of a row
 * @param[out] encoded_row_length the smallest row length cuvsFp8QuantizerTransform accepts for rows
 * of `dim` components
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsFp8QuantizerEncodedRowLength(uint32_t dim,
                                                         uint32_t* encoded_row_length);

/**
 * @brief Encode each row as FP8 (E4M3) codes with one float scale per row.
 *
 * A row's scale is its largest magnitude divided by 448, the largest finite E4M3 value, or 1 for an
 * all-zero row. Each component divided by the scale becomes the nearest E4M3 value among codes 0 to
 * 126 (127 is NaN), a tie going to the smaller magnitude, with the sign in bit 7 (0x80). Bytes past
 * the dimension are 0, and the full width of each row of `codes` is written. Component `j` of row
 * `i` decodes as `e4m3(codes[i, j]) * row_scales[i]`. The dataset values must be finite.
 *
 * These are the codes cuvsCagraUpdateScanCodes takes.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] dataset row-major [n_rows, dim] float32 or float16 matrix on the device
 * @param[out] codes row-major [n_rows, row_length] uint8 matrix on the device; row_length is a
 * multiple of 16 and at least cuvsFp8QuantizerEncodedRowLength(dim)
 * @param[out] row_scales [n_rows] float32 vector on the device
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsFp8QuantizerTransform(cuvsResources_t res,
                                                  DLManagedTensor* dataset,
                                                  DLManagedTensor* codes,
                                                  DLManagedTensor* row_scales);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
