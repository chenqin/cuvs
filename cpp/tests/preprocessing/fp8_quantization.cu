/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include <cuvs/preprocessing/quantize/fp8.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::preprocessing::quantize::fp8 {
namespace {

// The E4M3 rule on the host: the value of a code (0 to 126; 127 is NaN), and the code of a value,
// which is the nearest value among codes 0 to 126, a tie going to the smaller magnitude, with bit 7
// set for a negative value.
float e4m3_value(int c)
{
  int exponent = c >> 3, mantissa = c & 7;
  return exponent == 0
           ? static_cast<float>(mantissa) / 8.0f * std::ldexp(1.0f, -6)
           : (1.0f + static_cast<float>(mantissa) / 8.0f) * std::ldexp(1.0f, exponent - 7);
}

uint8_t host_code(float x)
{
  float a = std::fabs(x);
  int i   = 0;
  while (i < 127 && e4m3_value(i) < a) {
    i++;
  }
  i           = std::clamp(i, 1, 126);
  float below = e4m3_value(i - 1), above = e4m3_value(i);
  return static_cast<uint8_t>((a - below <= above - a ? i - 1 : i) | (x < 0.0f ? 0x80 : 0));
}

template <typename T>
auto from_float(float x) -> T
{
  if constexpr (std::is_same_v<T, half>) {
    return __float2half(x);
  } else {
    return x;
  }
}

template <typename T>
auto to_float(T x) -> float
{
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(x);
  } else {
    return x;
  }
}

}  // namespace

static_assert(get_encoded_row_length(0) == 0);
static_assert(get_encoded_row_length(1) == 16);
static_assert(get_encoded_row_length(15) == 16);
static_assert(get_encoded_row_length(16) == 16);
static_assert(get_encoded_row_length(17) == 32);
static_assert(get_encoded_row_length(1000) == 1008);

template <typename T>
struct Fp8QuantizationInputs {
  int64_t rows;
  int64_t dim;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Fp8QuantizationInputs<T>& inputs)
{
  return os << "rows:" << inputs.rows << " dim:" << inputs.dim;
}

template <typename T>
class Fp8QuantizationTest : public ::testing::TestWithParam<Fp8QuantizationInputs<T>> {
 public:
  Fp8QuantizationTest()
    : stream_(raft::resource::get_cuda_stream(handle_).get()),
      params_(::testing::TestWithParam<Fp8QuantizationInputs<T>>::GetParam()),
      rows_(params_.rows),
      dim_(params_.dim),
      dataset_(raft::make_device_matrix<T, int64_t>(handle_, rows_, dim_))
  {
  }

 protected:
  void SetUp() override
  {
    host_dataset_.resize(rows_ * dim_);
    std::mt19937 gen(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> exponent(-8, 8);
    for (int64_t i = 0; i < rows_; i++) {
      // Every row has its own magnitude, so the rows get different scales.
      const float magnitude = std::ldexp(1.0f, exponent(gen));
      for (int64_t d = 0; d < dim_; d++) {
        host_dataset_[i * dim_ + d] = from_float<T>(normal(gen) * magnitude);
      }
    }
    add_known_rows();
    if (!host_dataset_.empty()) {
      raft::update_device(
        dataset_.data_handle(), host_dataset_.data(), host_dataset_.size(), stream_);
    }
    raft::resource::sync_stream(handle_);
  }

  void test_transform()
  {
    const int64_t row_length = encoded_row_length();
    auto codes               = make_codes(rows_, row_length);
    auto row_scales          = make_row_scales(rows_);
    cuvs::execute_with_dry_run_check(
      handle_,
      [&](raft::resources const& res) {
        fp8::transform(res, dataset(), codes.view(), row_scales.view());
      },
      cuvs::alloc_behavior::NO_ALLOCATIONS);
    const auto got_codes  = to_host(codes.data_handle(), codes.size());
    const auto got_scales = to_host(row_scales.data_handle(), row_scales.size());
    EXPECT_TRUE(matches_reference(got_codes, row_length, got_scales));
    EXPECT_TRUE(matches_known_rows(got_codes, row_length, got_scales));
  }

  // Any multiple of 16 at least the encoded row length is accepted, and every byte past the
  // dimension is written as 0.
  void test_wider_row()
  {
    const int64_t row_length = encoded_row_length() + 48;
    auto codes               = make_codes(rows_, row_length);
    auto row_scales          = make_row_scales(rows_);
    fp8::transform(handle_, dataset(), codes.view(), row_scales.view());
    const auto got_codes  = to_host(codes.data_handle(), codes.size());
    const auto got_scales = to_host(row_scales.data_handle(), row_scales.size());
    EXPECT_TRUE(matches_reference(got_codes, row_length, got_scales));
    EXPECT_TRUE(matches_known_rows(got_codes, row_length, got_scales));
  }

  // The overload that fills a quantizer writes what the overload that takes views writes.
  void test_quantizer()
  {
    const int64_t row_length = encoded_row_length();
    auto codes               = make_codes(rows_, row_length);
    auto row_scales          = make_row_scales(rows_);
    fp8::transform(handle_, dataset(), codes.view(), row_scales.view());

    quantizer q(handle_, rows_, static_cast<uint32_t>(dim_));
    ASSERT_EQ(q.codes.extent(0), rows_);
    ASSERT_EQ(q.codes.extent(1), row_length);
    ASSERT_EQ(q.row_scales.extent(0), rows_);
    fill(q.codes.data_handle(), q.codes.size());
    fill(q.row_scales.data_handle(), q.row_scales.size());
    cuvs::execute_with_dry_run_check(
      handle_,
      [&](raft::resources const& res) { fp8::transform(res, dataset(), q); },
      cuvs::alloc_behavior::NO_ALLOCATIONS);

    const quantizer_view view = q.view();
    EXPECT_EQ(view.n_rows(), rows_);
    EXPECT_EQ(view.encoded_row_length(), static_cast<uint32_t>(row_length));
    EXPECT_EQ(view.codes.data_handle(), q.codes.data_handle());
    EXPECT_EQ(view.row_scales.data_handle(), q.row_scales.data_handle());
    EXPECT_EQ(view.row_scales.extent(0), rows_);
    EXPECT_EQ(to_host(view.codes.data_handle(), view.codes.size()),
              to_host(codes.data_handle(), codes.size()));
    EXPECT_EQ(to_host(view.row_scales.data_handle(), view.row_scales.size()),
              to_host(row_scales.data_handle(), row_scales.size()));
  }

  void test_bad_extents()
  {
    const auto dim           = static_cast<uint32_t>(dim_);
    const int64_t row_length = encoded_row_length();
    auto codes       = raft::make_device_matrix<uint8_t, int64_t>(handle_, rows_, row_length);
    auto row_scales  = raft::make_device_vector<float, int64_t>(handle_, rows_);
    auto more_rows   = raft::make_device_matrix<uint8_t, int64_t>(handle_, rows_ + 1, row_length);
    auto too_narrow  = raft::make_device_matrix<uint8_t, int64_t>(handle_, rows_, row_length - 16);
    auto unaligned   = raft::make_device_matrix<uint8_t, int64_t>(handle_, rows_, row_length + 8);
    auto more_scales = raft::make_device_vector<float, int64_t>(handle_, rows_ + 1);
    EXPECT_THROW(fp8::transform(handle_, dataset(), more_rows.view(), row_scales.view()),
                 raft::logic_error);
    EXPECT_THROW(fp8::transform(handle_, dataset(), too_narrow.view(), row_scales.view()),
                 raft::logic_error);
    EXPECT_THROW(fp8::transform(handle_, dataset(), unaligned.view(), row_scales.view()),
                 raft::logic_error);
    EXPECT_THROW(fp8::transform(handle_, dataset(), codes.view(), more_scales.view()),
                 raft::logic_error);

    quantizer more_rows_q(handle_, rows_ + 1, dim);
    EXPECT_THROW(fp8::transform(handle_, dataset(), more_rows_q), raft::logic_error);
    if (dim > 16) {
      quantizer too_narrow_q(handle_, rows_, dim - 16);
      EXPECT_THROW(fp8::transform(handle_, dataset(), too_narrow_q), raft::logic_error);
    }

    EXPECT_NO_THROW(fp8::transform(handle_, dataset(), codes.view(), row_scales.view()));
    raft::resource::sync_stream(handle_);
  }

 private:
  struct known_code {
    int64_t row;
    int64_t component;
    uint8_t code;
  };

  [[nodiscard]] auto encoded_row_length() const -> int64_t
  {
    return static_cast<int64_t>(get_encoded_row_length(static_cast<uint32_t>(dim_)));
  }

  [[nodiscard]] auto dataset() const -> raft::device_matrix_view<const T, int64_t, raft::row_major>
  {
    return dataset_.view();
  }

  // Bytes the transform leaves alone keep this pattern, so a skipped byte shows.
  template <typename U>
  void fill(U* data, size_t n)
  {
    if (n > 0) { RAFT_CUDA_TRY(cudaMemsetAsync(data, 0xff, n * sizeof(U), stream_)); }
  }

  auto make_codes(int64_t rows, int64_t row_length) -> raft::device_matrix<uint8_t, int64_t>
  {
    auto codes = raft::make_device_matrix<uint8_t, int64_t>(handle_, rows, row_length);
    fill(codes.data_handle(), codes.size());
    return codes;
  }

  auto make_row_scales(int64_t rows) -> raft::device_vector<float, int64_t>
  {
    auto row_scales = raft::make_device_vector<float, int64_t>(handle_, rows);
    fill(row_scales.data_handle(), row_scales.size());
    return row_scales;
  }

  template <typename U>
  auto to_host(const U* data, size_t n) -> std::vector<U>
  {
    std::vector<U> out(n);
    if (n > 0) { raft::update_host(out.data(), data, n, stream_); }
    raft::resource::sync_stream(handle_);
    return out;
  }

  [[nodiscard]] auto value(int64_t row, int64_t component) const -> float
  {
    return to_float(host_dataset_[row * dim_ + component]);
  }

  [[nodiscard]] auto reference_scale(int64_t row) const -> float
  {
    float amax = 0.0f;
    for (int64_t d = 0; d < dim_; d++) {
      amax = std::max(amax, std::fabs(value(row, d)));
    }
    return amax > 0.0f ? amax / 448.0f : 1.0f;
  }

  // Every scale and every byte of the codes, row_length bytes a row, against the host rule.
  [[nodiscard]] auto matches_reference(const std::vector<uint8_t>& codes,
                                       int64_t row_length,
                                       const std::vector<float>& row_scales) const
    -> ::testing::AssertionResult
  {
    for (int64_t i = 0; i < rows_; i++) {
      const float scale = reference_scale(i);
      if (row_scales[i] != scale) {
        return ::testing::AssertionFailure()
               << "row " << i << ": scale " << row_scales[i] << ", expected " << scale;
      }
      for (int64_t d = 0; d < row_length; d++) {
        const uint8_t want = d < dim_ ? host_code(value(i, d) / scale) : 0;
        const uint8_t got  = codes[i * row_length + d];
        if (got != want) {
          return ::testing::AssertionFailure() << "row " << i << " byte " << d << ": code "
                                               << int(got) << ", expected " << int(want);
        }
      }
    }
    return ::testing::AssertionSuccess();
  }

  [[nodiscard]] auto matches_known_rows(const std::vector<uint8_t>& codes,
                                        int64_t row_length,
                                        const std::vector<float>& row_scales) const
    -> ::testing::AssertionResult
  {
    for (const auto& [row, scale] : known_scales_) {
      if (row_scales[row] != scale) {
        return ::testing::AssertionFailure()
               << "row " << row << ": scale " << row_scales[row] << ", expected " << scale;
      }
    }
    for (const auto& known : known_codes_) {
      const uint8_t got = codes[known.row * row_length + known.component];
      if (got != known.code) {
        return ::testing::AssertionFailure()
               << "row " << known.row << " component " << known.component << " (value "
               << value(known.row, known.component) << "): code " << int(got) << ", expected "
               << int(known.code);
      }
    }
    return ::testing::AssertionSuccess();
  }

  // Sets one component, and the code it must get when code >= 0.
  void set(int64_t row, int64_t component, float x, int code = -1)
  {
    host_dataset_[row * dim_ + component] = from_float<T>(x);
    if (code >= 0) { known_codes_.push_back({row, component, static_cast<uint8_t>(code)}); }
  }

  // Rows whose codes are known without the reference (all values are exact in half as well).
  void add_known_rows()
  {
    const int64_t slots = dim_ - 1;  // the components after the first
    // Each code and each midpoint twice: once positive, once negative.
    const int64_t exact_rows = slots > 0 ? (2 * 127 + slots - 1) / slots : 0;
    const int64_t tie_rows   = slots > 0 ? (2 * 126 + slots - 1) / slots : 0;
    if (dim_ < 1 || rows_ < 4 + exact_rows + tie_rows) { return; }
    int64_t row = 0;

    // An all-zero row has scale 1 and only zero codes; so has a row of negative zeros, as a zero
    // has no sign bit.
    for (const float zero : {0.0f, -0.0f}) {
      for (int64_t d = 0; d < dim_; d++) {
        set(row, d, zero, 0x00);
      }
      known_scales_.emplace_back(row++, 1.0f);
    }

    // The largest magnitude gets the largest code, 0x7e, or 0xfe when it is negative.
    for (const float amax : {3.0f, -3.0f}) {
      set(row, 0, amax, amax > 0.0f ? 0x7e : 0xfe);
      for (int64_t d = 1; d < dim_; d++) {
        if (d == 1) {
          set(row, d, -amax, amax > 0.0f ? 0xfe : 0x7e);
        } else {
          set(row, d, static_cast<float>(d % 7 - 3) * 0.25f);
        }
      }
      known_scales_.emplace_back(row++, 3.0f / 448.0f);
    }

    // With 448 first the scale is 1: every E4M3 value (subnormal ones included) gets its own code,
    // and the midpoint between two neighbouring values gets the smaller code.
    for (int64_t r = 0; r < exact_rows; r++, row++) {
      set(row, 0, 448.0f, 0x7e);
      for (int64_t d = 1; d < dim_; d++) {
        const int64_t k     = r * slots + d - 1;
        const int c         = static_cast<int>(k % 127);
        const bool negative = (k / 127) % 2 == 1;
        set(row, d, negative ? -e4m3_value(c) : e4m3_value(c), c | (negative && c > 0 ? 0x80 : 0));
      }
      known_scales_.emplace_back(row, 1.0f);
    }
    for (int64_t r = 0; r < tie_rows; r++, row++) {
      set(row, 0, -448.0f, 0xfe);
      for (int64_t d = 1; d < dim_; d++) {
        const int64_t k      = r * slots + d - 1;
        const int c          = static_cast<int>(k % 126);
        const bool negative  = (k / 126) % 2 == 1;
        const float midpoint = (e4m3_value(c) + e4m3_value(c + 1)) / 2.0f;
        set(row, d, negative ? -midpoint : midpoint, c | (negative ? 0x80 : 0));
      }
      known_scales_.emplace_back(row, 1.0f);
    }
  }

  raft::resources handle_;
  cudaStream_t stream_;
  Fp8QuantizationInputs<T> params_;
  int64_t rows_;
  int64_t dim_;
  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  std::vector<T> host_dataset_;
  std::vector<known_code> known_codes_;
  std::vector<std::pair<int64_t, float>> known_scales_;
};

template <typename T>
auto generate_inputs() -> std::vector<Fp8QuantizationInputs<T>>
{
  return raft::util::itertools::product<Fp8QuantizationInputs<T>>(
    {0, 1, 2000},                     // rows
    {1, 15, 16, 17, 33, 128, 1000});  // dim
}

using Fp8QuantizationTest_float = Fp8QuantizationTest<float>;
TEST_P(Fp8QuantizationTest_float, TransformMatchesReference) { this->test_transform(); }
TEST_P(Fp8QuantizationTest_float, WiderRowZeroFilled) { this->test_wider_row(); }
TEST_P(Fp8QuantizationTest_float, QuantizerMatchesViews) { this->test_quantizer(); }
TEST_P(Fp8QuantizationTest_float, RejectsBadExtents) { this->test_bad_extents(); }

using Fp8QuantizationTest_half = Fp8QuantizationTest<half>;
TEST_P(Fp8QuantizationTest_half, TransformMatchesReference) { this->test_transform(); }
TEST_P(Fp8QuantizationTest_half, WiderRowZeroFilled) { this->test_wider_row(); }
TEST_P(Fp8QuantizationTest_half, QuantizerMatchesViews) { this->test_quantizer(); }
TEST_P(Fp8QuantizationTest_half, RejectsBadExtents) { this->test_bad_extents(); }

INSTANTIATE_TEST_CASE_P(Fp8QuantizationTest,
                        Fp8QuantizationTest_float,
                        ::testing::ValuesIn(generate_inputs<float>()));
INSTANTIATE_TEST_CASE_P(Fp8QuantizationTest,
                        Fp8QuantizationTest_half,
                        ::testing::ValuesIn(generate_inputs<half>()));

}  // namespace cuvs::preprocessing::quantize::fp8
