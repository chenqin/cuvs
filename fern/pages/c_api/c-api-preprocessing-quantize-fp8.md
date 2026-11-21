---
slug: api-reference/c-api-preprocessing-quantize-fp8
---

# Fp8

_Source header: `cuvs/preprocessing/quantize/fp8.h`_

## C API for FP8 (E4M3) Quantizer

<a id="cuvsfp8quantizerencodedrowlength"></a>
### cuvsFp8QuantizerEncodedRowLength

Bytes of one row of codes: the dimension rounded up to a multiple of 16.

```c
cuvsError_t cuvsFp8QuantizerEncodedRowLength(uint32_t dim,
uint32_t* encoded_row_length);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` | in | `uint32_t` | number of components of a row |
| `encoded_row_length` | out | `uint32_t*` | the smallest row length cuvsFp8QuantizerTransform accepts for rows of `dim` components |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsfp8quantizertransform"></a>
### cuvsFp8QuantizerTransform

Encode each row as FP8 (E4M3) codes with one float scale per row.

```c
cuvsError_t cuvsFp8QuantizerTransform(cuvsResources_t res,
DLManagedTensor* dataset,
DLManagedTensor* codes,
DLManagedTensor* row_scales);
```

A row's scale is its largest magnitude divided by 448, the largest finite E4M3 value, or 1 for an all-zero row. Each component divided by the scale becomes the nearest E4M3 value among codes 0 to 126 (127 is NaN), a tie going to the smaller magnitude, with the sign in bit 7 (0x80). Bytes past the dimension are 0, and the full width of each row of `codes` is written. Component `j` of row `i` decodes as `e4m3(codes[i, j]) * row_scales[i]`. The dataset values must be finite.

These are the codes cuvsCagraUpdateScanCodes takes.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `dataset` | in | `DLManagedTensor*` | row-major [n_rows, dim] float32 or float16 matrix on the device |
| `codes` | out | `DLManagedTensor*` | row-major [n_rows, row_length] uint8 matrix on the device; row_length is a multiple of 16 and at least cuvsFp8QuantizerEncodedRowLength(dim) |
| `row_scales` | out | `DLManagedTensor*` | [n_rows] float32 vector on the device |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
