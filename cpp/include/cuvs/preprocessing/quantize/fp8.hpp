/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>

#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace fp8 {

/**
 * @defgroup fp8 FP8 (E4M3) row-scaled quantizer
 * @{
 */

/**
 * @brief Bytes of one row of codes: the dimension rounded up to a multiple of 16, so that each 16
 * components are one aligned 16-byte load.
 *
 * @param[in] dim number of components of a row
 * @return the smallest row length `transform` accepts for rows of @p dim components
 */
[[nodiscard]] constexpr auto get_encoded_row_length(uint32_t dim) noexcept -> uint32_t
{
  return raft::div_rounding_up_safe(dim, 16u) * 16u;
}

/**
 * @brief Non-owning view of FP8 (E4M3) codes with one scale per row.
 *
 * Component `j` of row `i` decodes as `e4m3(codes(i, j)) * row_scales(i)`; bytes past the
 * dimension are 0. The rows of the CAGRA filtered scan use this layout
 * (`cuvs::neighbors::cagra::index::update_scan_codes`).
 */
struct quantizer_view {
  /** [n_rows, encoded_row_length] E4M3 codes on the device, one row per dataset row. */
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> codes;
  /** [n_rows] scale of each row on the device. */
  raft::device_vector_view<const float, int64_t> row_scales;

  /** Number of encoded rows. */
  [[nodiscard]] constexpr auto n_rows() const noexcept -> int64_t { return codes.extent(0); }
  /** Bytes of one row of codes (a multiple of 16). */
  [[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(codes.extent(1));
  }
};

/**
 * @brief FP8 (E4M3) codes with one scale per row, owned on the device.
 *
 * The constructor only allocates; `transform` fills the codes and the scales.
 *
 * Usage example:
 * @code{.cpp}
 * namespace fp8 = cuvs::preprocessing::quantize::fp8;
 * raft::resources res;
 * // dataset: a [n_rows, dim] row-major device matrix view of float or half
 * fp8::quantizer q(res, n_rows, dim);
 * fp8::transform(res, dataset, q);
 * fp8::quantizer_view codes = q.view();
 * @endcode
 */
struct quantizer {
  /** [n_rows, get_encoded_row_length(dim)] E4M3 codes on the device. */
  raft::device_matrix<uint8_t, int64_t, raft::row_major> codes;
  /** [n_rows] scale of each row on the device. */
  raft::device_vector<float, int64_t> row_scales;

  /**
   * @brief Allocate (uninitialized) codes and scales for @p n_rows rows of @p dim components.
   *
   * @param[in] res raft resources
   * @param[in] n_rows number of rows
   * @param[in] dim number of components of a row
   */
  quantizer(raft::resources const& res, int64_t n_rows, uint32_t dim);

  /** A non-owning view of the codes and the scales. */
  [[nodiscard]] auto view() const noexcept -> quantizer_view
  {
    return quantizer_view{codes.view(), row_scales.view()};
  }
};

/**
 * @brief Encode each row as FP8 (E4M3) codes with one float scale per row.
 *
 * A row's scale is its largest magnitude divided by 448, the largest finite E4M3 value, or 1 for an
 * all-zero row. Each component divided by the scale becomes the nearest E4M3 value among codes 0 to
 * 126 (127 is NaN), a tie going to the smaller magnitude, with the sign in bit 7 (0x80). Bytes past
 * the dimension are 0, and the full width of each row of @p codes is written. Component `j` of row
 * `i` decodes as `e4m3(codes(i, j)) * row_scales(i)`.
 *
 * `__nv_cvt_float_to_fp8` rounds ties to even instead, so its codes can differ from these at the
 * exact midpoints between two E4M3 values.
 *
 * FP8 row scaling has nothing to train: each row's scale is an output, so these overloads take no
 * trained quantizer (unlike binary::transform, whose quantizer-less overloads are deprecated in
 * favour of a trained threshold). `quantizer` only owns the outputs.
 *
 * Usage example:
 * @code{.cpp}
 * namespace fp8 = cuvs::preprocessing::quantize::fp8;
 * raft::resources res;
 * // dataset: a [n_rows, dim] row-major device matrix view of float or half
 * auto codes = raft::make_device_matrix<uint8_t, int64_t>(
 *   res, n_rows, fp8::get_encoded_row_length(dim));
 * auto row_scales = raft::make_device_vector<float, int64_t>(res, n_rows);
 * fp8::transform(res, dataset, codes.view(), row_scales.view());
 * @endcode
 *
 * @pre The dataset values are finite.
 *
 * @param[in] res raft resources
 * @param[in] dataset [n_rows, dim] rows on the device
 * @param[out] codes [n_rows, row_length] on the device; row_length is a multiple of 16 and at least
 * get_encoded_row_length(dim)
 * @param[out] row_scales [n_rows] on the device
 */
void transform(raft::resources const& res,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> row_scales);

/** @copydoc transform */
void transform(raft::resources const& res,
               raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> row_scales);

/**
 * @brief Encode each row into the codes and scales a quantizer owns.
 *
 * The same encoding as the overload that takes the codes and the scales as views; the quantizer
 * must have as many rows as the dataset and a row length of at least
 * get_encoded_row_length(dim).
 *
 * Usage example:
 * @code{.cpp}
 * namespace fp8 = cuvs::preprocessing::quantize::fp8;
 * fp8::quantizer q(res, n_rows, dim);
 * fp8::transform(res, dataset, q);
 * @endcode
 *
 * @pre The dataset values are finite.
 *
 * @param[in] res raft resources
 * @param[in] dataset [n_rows, dim] rows on the device
 * @param[out] out codes and scales of the rows
 */
void transform(raft::resources const& res,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               quantizer& out);

/**
 * @brief Encode each row into the codes and scales a quantizer owns.
 *
 * The same encoding as the overload that takes the codes and the scales as views; the quantizer
 * must have as many rows as the dataset and a row length of at least
 * get_encoded_row_length(dim).
 *
 * @pre The dataset values are finite.
 *
 * @param[in] res raft resources
 * @param[in] dataset [n_rows, dim] rows on the device
 * @param[out] out codes and scales of the rows
 */
void transform(raft::resources const& res,
               raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
               quantizer& out);

/** @} */  // end of group fp8

}  // namespace fp8
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
