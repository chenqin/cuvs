---
slug: api-reference/cpp-api-preprocessing-quantize-fp8
---

# Fp8

_Source header: `cuvs/preprocessing/quantize/fp8.hpp`_

## FP8 (E4M3) row-scaled quantizer

<a id="preprocessing-quantize-fp8-get-encoded-row-length"></a>
### preprocessing::quantize::fp8::get_encoded_row_length

Bytes of one row of codes: the dimension rounded up to a multiple of 16, so that each 16 components are one aligned 16-byte load.

```cpp
[[nodiscard]] constexpr auto get_encoded_row_length(uint32_t dim) noexcept -> uint32_t;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` | in | `uint32_t` | number of components of a row |

**Returns**

`uint32_t`

the smallest row length `transform` accepts for rows of `dim` components

<a id="preprocessing-quantize-fp8-quantizer-view"></a>
### preprocessing::quantize::fp8::quantizer_view

Non-owning view of FP8 (E4M3) codes with one scale per row.

Component `j` of row `i` decodes as `e4m3(codes(i, j)) * row_scales(i)`; bytes past the dimension are 0. The rows of the CAGRA filtered scan use this layout (`cuvs::neighbors::cagra::index::update_scan_codes`).

```cpp
struct quantizer_view {
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> codes;
  raft::device_vector_view<const float, int64_t> row_scales;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `codes` | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | [n_rows, encoded_row_length] E4M3 codes on the device, one row per dataset row. |
| `row_scales` | `raft::device_vector_view<const float, int64_t>` | [n_rows] scale of each row on the device. |

<a id="preprocessing-quantize-fp8-quantizer-view-n-rows"></a>
### preprocessing::quantize::fp8::quantizer_view::n_rows

Number of encoded rows.

```cpp
[[nodiscard]] constexpr auto n_rows() const noexcept -> int64_t;
```

**Returns**

`int64_t`

<a id="preprocessing-quantize-fp8-quantizer-view-encoded-row-length"></a>
### preprocessing::quantize::fp8::quantizer_view::encoded_row_length

Bytes of one row of codes (a multiple of 16).

```cpp
[[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="preprocessing-quantize-fp8-quantizer"></a>
### preprocessing::quantize::fp8::quantizer

FP8 (E4M3) codes with one scale per row, owned on the device.

The constructor only allocates; `transform` fills the codes and the scales.

Usage example:

```cpp
struct quantizer {
  raft::device_matrix<uint8_t, int64_t, raft::row_major> codes;
  raft::device_vector<float, int64_t> row_scales;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `codes` | `raft::device_matrix<uint8_t, int64_t, raft::row_major>` | [n_rows, get_encoded_row_length(dim)] E4M3 codes on the device. |
| `row_scales` | `raft::device_vector<float, int64_t>` | [n_rows] scale of each row on the device. |

<a id="preprocessing-quantize-fp8-quantizer-quantizer"></a>
### preprocessing::quantize::fp8::quantizer::quantizer

Allocate (uninitialized) codes and scales for `n_rows` rows of `dim` components.

```cpp
quantizer(raft::resources const& res, int64_t n_rows, uint32_t dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `n_rows` | in | `int64_t` | number of rows |
| `dim` | in | `uint32_t` | number of components of a row |

**Returns**

`void`

<a id="preprocessing-quantize-fp8-quantizer-view"></a>
### preprocessing::quantize::fp8::quantizer::view

A non-owning view of the codes and the scales.

```cpp
[[nodiscard]] auto view() const noexcept -> quantizer_view;
```

**Returns**

[`quantizer_view`](/api-reference/cpp-api-preprocessing-quantize-fp8#preprocessing-quantize-fp8-quantizer-view)

<a id="preprocessing-quantize-fp8-transform"></a>
### preprocessing::quantize::fp8::transform

Encode each row as FP8 (E4M3) codes with one float scale per row.

```cpp
void transform(raft::resources const& res,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> row_scales);
```

A row's scale is its largest magnitude divided by 448, the largest finite E4M3 value, or 1 for an all-zero row. Each component divided by the scale becomes the nearest E4M3 value among codes 0 to 126 (127 is NaN), a tie going to the smaller magnitude, with the sign in bit 7 (0x80). Bytes past the dimension are 0, and the full width of each row of `codes` is written. Component `j` of row `i` decodes as `e4m3(codes(i, j)) * row_scales(i)`.

`__nv_cvt_float_to_fp8` rounds ties to even instead, so its codes can differ from these at the exact midpoints between two E4M3 values.

FP8 row scaling has nothing to train: each row's scale is an output, so these overloads take no trained quantizer (unlike binary::transform, whose quantizer-less overloads are deprecated in favour of a trained threshold). `quantizer` only owns the outputs.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | [n_rows, dim] rows on the device |
| `codes` | out | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` | [n_rows, row_length] on the device; row_length is a multiple of 16 and at least get_encoded_row_length(dim) |
| `row_scales` | out | `raft::device_vector_view<float, int64_t>` | [n_rows] on the device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::fp8::transform`

```cpp
void transform(raft::resources const& res,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> row_scales);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `codes` |  | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` |  |
| `row_scales` |  | `raft::device_vector_view<float, int64_t>` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::fp8::transform`

Encode each row into the codes and scales a quantizer owns.

```cpp
void transform(raft::resources const& res,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
quantizer& out);
```

The same encoding as the overload that takes the codes and the scales as views; the quantizer must have as many rows as the dataset and a row length of at least get_encoded_row_length(dim).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | [n_rows, dim] rows on the device |
| `out` | out | [`quantizer&`](/api-reference/cpp-api-preprocessing-quantize-fp8#preprocessing-quantize-fp8-quantizer) | codes and scales of the rows |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::fp8::transform`

Encode each row into the codes and scales a quantizer owns.

```cpp
void transform(raft::resources const& res,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
quantizer& out);
```

The same encoding as the overload that takes the codes and the scales as views; the quantizer must have as many rows as the dataset and a row length of at least get_encoded_row_length(dim).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | [n_rows, dim] rows on the device |
| `out` | out | [`quantizer&`](/api-reference/cpp-api-preprocessing-quantize-fp8#preprocessing-quantize-fp8-quantizer) | codes and scales of the rows |

**Returns**

`void`
