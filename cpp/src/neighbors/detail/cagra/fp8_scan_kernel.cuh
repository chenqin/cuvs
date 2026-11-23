/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * The FP8 (E4M3, row-scaled) tensor-core scan of the filtered scan (fp8_scan.cu).
 *
 * Each warp takes 16 rows at a time, decodes their E4M3 bytes to FP16 with the hardware conversion
 * (cvt.rn.f16x2.e4m3x2) straight into mma.sync A fragments, and multiplies them against up to
 * 8 * NT queries held as FP16 in shared memory (FP32 accumulate). Distances follow from
 * |x|^2 + |q|^2 - 2 x.q, all scaled by the row scale. score_fp8 writes a [queries][rows] score
 * matrix; fuse_fp8 keeps only distances at or under each query's bound (bounds[q * 32 + 31]),
 * appending (distance, position in rows) to the query's candidate list.
 *
 * The conversion needs sm_89 or newer. Images for older architectures get empty scan kernels, and
 * search() runs only where is_supported() found an sm_89 or newer image.
 */

#include <raft/core/detail/macros.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::fp8_scan {

/** k-steps (16 dimensions each) loaded per round: 128 dimensions. */
inline constexpr int kSegSteps = 8;
/** Halves past each query's row in shared memory: moves the eight lanes reading one B fragment
 * off a single bank. */
inline constexpr int kQueryPad = 8;
/** Hits per query the fused scan stages in shared memory per block before appending them with one
 * global atomic; a hit past the stage takes the global path. */
inline constexpr int kStage = 32;
/** Entries of each lane's shared hit list in the fused scan: a tile's hits go there (two stores,
 * no atomics in the row loop) and are staged per query after the tile; a hit past the list appends
 * directly. The fused scan reads each query's norm and cutoff as one shared float2 (cutoff -inf
 * past nq). */
inline constexpr int kHitCap = 32;
/** Threads of a scan block; the fused scan's per-lane hit lists assume it. */
inline constexpr int kScanThreads = 256;

__device__ __forceinline__ float half_bits(unsigned short h)
{
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
/* Two E4M3 bytes (the low 16 bits of word) as two FP16 values. */
__device__ __forceinline__ unsigned e4m3_pair(unsigned word)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  unsigned h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((unsigned short)(word & 0xffffu)));
  return h2;
#else
  return 0u;
#endif
}
__device__ __forceinline__ float sumsq2(unsigned h2)
{
  float lo = half_bits((unsigned short)(h2 & 0xffffu)), hi = half_bits((unsigned short)(h2 >> 16));
  return lo * lo + hi * hi;
}
/* D += A (16 rows x 16 k, FP16) * B (16 k x 8 queries, FP16), FP32 accumulate. */
__device__ __forceinline__ void mma_m16n8k16(
  float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
    "{%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
/* Bytes of the half queries in a scan kernel's dynamic shared memory, rounded up to 16. */
__host__ __device__ constexpr unsigned fp8_query_bytes(unsigned queries, unsigned qstride)
{
  return (queries * qstride * 2 + 15) & ~15u;
}
/* Bytes of the fused scan's per-lane hit lists after them. */
constexpr unsigned fp8_hit_bytes = kHitCap * kScanThreads * (sizeof(float) + 1);
/* Rows at or past n_codes (the list's end included) are skipped: +inf in the matrix, never kept. */
template <int NT, bool Fused>
__device__ void score_mma_fp8(const unsigned char* data,
                              const float* row_scales,
                              unsigned row_bytes,
                              unsigned blocks,
                              unsigned n_codes,
                              const unsigned* rows,
                              unsigned n,
                              const float* queries,
                              int nq,
                              int dims,
                              float* out,
                              const float* bounds,
                              unsigned* cand_r,
                              unsigned* counts,
                              unsigned cap)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  /* Dynamic shared memory: the half queries (fp8_query_bytes), then for the fused scan each lane's
   * hit list (kHitCap distances and tile slots a lane, fp8_hit_bytes). The lists are dynamic so
   * that the static shared memory stays within 48 KB. */
  extern __shared__ __align__(16) unsigned char fp8_smem[];
  unsigned short* qh = (unsigned short*)fp8_smem;
  __shared__ float qn[NT * 8];
  __shared__ float2 qt[Fused ? NT * 8 : 1];
  int qstride          = ((dims + 15) & ~15) + kQueryPad;
  float* hit_d         = (float*)(fp8_smem + fp8_query_bytes(NT * 8, qstride));
  unsigned char* hit_j = (unsigned char*)(hit_d + kHitCap * kScanThreads);
  unsigned hit_count   = 0;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  int g = lane >> 2, t = lane & 3;
  const float inf = __int_as_float(0x7f800000);
  __shared__ float stage_d[Fused ? NT * 8 * kStage : 1];
  __shared__ unsigned stage_r[Fused ? NT * 8 * kStage : 1];
  __shared__ unsigned stage_count[NT * 8];
  if (threadIdx.x < NT * 8) stage_count[threadIdx.x] = 0;
  auto append = [&](unsigned q, float d, unsigned r) {
    unsigned at = atomicAdd(stage_count + q, 1u);
    if (at < kStage) {
      stage_d[q * kStage + at] = d;
      stage_r[q * kStage + at] = r;
    } else {
      at = atomicAdd(counts + q, 1u);
      if (at < cap) {
        out[(unsigned long long)q * cap + at]    = d;
        cand_r[(unsigned long long)q * cap + at] = r;
      }
    }
  };
  /* The half queries are staged four floats per load, the loads of a thread unrolled. */
  if ((dims & 3) == 0 && (qstride & 3) == 0) {
    int q4 = qstride >> 2;
#pragma unroll 4
    for (int i = threadIdx.x; i < NT * 8 * q4; i += blockDim.x) {
      int q = i / q4, d = (i - q * q4) << 2;
      float4 v = (q < nq && d < dims) ? *(const float4*)(queries + q * dims + d)
                                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      unsigned short h0, h1, h2, h3;
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h0) : "f"(v.x));
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h1) : "f"(v.y));
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h2) : "f"(v.z));
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h3) : "f"(v.w));
      *(uint2*)(qh + 4 * i) =
        make_uint2((unsigned)h0 | ((unsigned)h1 << 16), (unsigned)h2 | ((unsigned)h3 << 16));
    }
  } else
    for (int i = threadIdx.x; i < NT * 8 * qstride; i += blockDim.x) {
      int q = i / qstride, d = i - q * qstride;
      float v = (q < nq && d < dims) ? queries[q * dims + d] : 0.0f;
      unsigned short h;
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
      qh[i] = h;
    }
  __syncthreads();
  if (threadIdx.x < NT * 8) {
    float s = 0.0f;
    for (int d = 0; d < dims; d++) {
      float v = half_bits(qh[threadIdx.x * qstride + d]);
      s += v * v;
    }
    qn[threadIdx.x] = s;
    if (Fused)
      qt[threadIdx.x] = make_float2(s, threadIdx.x < nq ? bounds[threadIdx.x * 32 + 31] : -inf);
  }
  __syncthreads();
  unsigned stride = gridDim.x * warps * 16, first = (blockIdx.x * warps + warp) * 16;
  for (unsigned base = first; base < n; base += stride) {
    unsigned i0 = base + g, i1 = base + g + 8;
    unsigned row0 = i0 < n ? rows[i0] : 0xffffffffu;
    unsigned row1 = i1 < n ? rows[i1] : 0xffffffffu;
    bool v0 = row0 < n_codes, v1 = row1 < n_codes;
    unsigned r0 = v0 ? row0 : 0u, r1 = v1 ? row1 : 0u;
    float acc[NT * 4];
    float nx0 = 0.0f, nx1 = 0.0f;
#pragma unroll
    for (int j = 0; j < NT * 4; j++)
      acc[j] = 0.0f;
    for (unsigned seg = 0; seg < blocks; seg += kSegSteps) {
      uint2 w0[kSegSteps], w1[kSegSteps];
      const uint4* p0 = (const uint4*)(data + (unsigned long long)r0 * row_bytes) + seg;
      const uint4* p1 = (const uint4*)(data + (unsigned long long)r1 * row_bytes) + seg;
#pragma unroll
      for (int k = 0; k < kSegSteps; k++)
        if (seg + k < blocks) {
          uint4 x0 = p0[k], x1 = p1[k];
          w0[k].x = t < 2 ? x0.x : x0.y;
          w0[k].y = t < 2 ? x0.z : x0.w;
          w1[k].x = t < 2 ? x1.x : x1.y;
          w1[k].y = t < 2 ? x1.z : x1.w;
        }
#pragma unroll
      for (int k = 0; k < kSegSteps; k++) {
        if (seg + k >= blocks) break;
        unsigned a0 = e4m3_pair(w0[k].x >> (16 * (t & 1))),
                 a2 = e4m3_pair(w0[k].y >> (16 * (t & 1)));
        unsigned a1 = e4m3_pair(w1[k].x >> (16 * (t & 1))),
                 a3 = e4m3_pair(w1[k].y >> (16 * (t & 1)));
        /* Dimensions past dims (padding) count for nothing. */
        if ((seg + k) * 16 + 16 > dims) {
          int d0 = (seg + k) * 16 + 2 * t, d2 = d0 + 8;
          unsigned m0 = d0 + 1 < dims ? 0xffffffffu : d0 < dims ? 0x0000ffffu : 0u;
          unsigned m2 = d2 + 1 < dims ? 0xffffffffu : d2 < dims ? 0x0000ffffu : 0u;
          a0 &= m0;
          a1 &= m0;
          a2 &= m2;
          a3 &= m2;
        }
        nx0 += sumsq2(a0) + sumsq2(a2);
        nx1 += sumsq2(a1) + sumsq2(a3);
        const unsigned short* qk = qh + (seg + k) * 16 + 2 * t;
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          const unsigned short* qq = qk + (nt * 8 + g) * qstride;
          unsigned b0 = *(const unsigned*)qq, b1 = *(const unsigned*)(qq + 8);
          mma_m16n8k16(acc + nt * 4, a0, a1, a2, a3, b0, b1);
        }
      }
    }
    nx0 += __shfl_xor_sync(0xffffffffu, nx0, 1);
    nx0 += __shfl_xor_sync(0xffffffffu, nx0, 2);
    nx1 += __shfl_xor_sync(0xffffffffu, nx1, 1);
    nx1 += __shfl_xor_sync(0xffffffffu, nx1, 2);
    float rs0 = row_scales[r0], rs1 = row_scales[r1];
    /* acc[nt * 4 + c]: c0 (row g, query nt*8 + 2t), c1 (g, +1), c2 (row g + 8, 2t),
     * c3 (g + 8, +1). */
#pragma unroll
    for (int j = 0; j < NT * 4; j++) {
      int nn     = (j >> 2) * 8 + 2 * t + (j & 1);
      unsigned i = (j & 2) ? i1 : i0;
      bool vr    = (j & 2) ? v1 : v0;
      float nx   = (j & 2) ? nx1 : nx0;
      float rs   = (j & 2) ? rs1 : rs0;
      float2 qv  = Fused ? qt[nn] : make_float2(qn[nn], 0.0f);
      float dist = fmaxf(rs * rs * nx + qv.x - 2.0f * rs * acc[j], 0.0f);
      if (Fused) {
        if (vr && dist <= qv.y) {
          if (kHitCap >= NT * 4 || hit_count < kHitCap) {
            hit_d[hit_count * kScanThreads + threadIdx.x] = dist;
            hit_j[hit_count * kScanThreads + threadIdx.x] = (unsigned char)j;
            hit_count++;
          } else
            append(nn, dist, i);
        }
      } else if (nn < nq && i < n)
        out[(unsigned long long)nn * n + i] = vr ? dist : inf;
    }
    if (Fused) {
      for (unsigned e = 0; e < hit_count; e++) {
        unsigned jj = hit_j[e * kScanThreads + threadIdx.x];
        append((jj >> 2) * 8 + 2 * t + (jj & 1),
               hit_d[e * kScanThreads + threadIdx.x],
               base + g + (jj & 2) * 4);
      }
      hit_count = 0;
    }
  }
  if (Fused) {
    __syncthreads();
    for (int q = threadIdx.x; q < NT * 8; q += blockDim.x) {
      unsigned c = stage_count[q] < kStage ? stage_count[q] : kStage;
      if (c == 0 || q >= nq) continue;
      unsigned base = atomicAdd(counts + q, c);
      for (unsigned j = 0; j < c && base + j < cap; j++) {
        out[(unsigned long long)q * cap + base + j]    = stage_d[q * kStage + j];
        cand_r[(unsigned long long)q * cap + base + j] = stage_r[q * kStage + j];
      }
    }
  }
#endif
}
/* A [nq][n] matrix of squared distances of the listed rows (rows at or past n_codes score +inf). */
template <int NT>
RAFT_KERNEL score_fp8(const unsigned char* data,
                      const float* row_scales,
                      unsigned row_bytes,
                      unsigned blocks,
                      unsigned n_codes,
                      const unsigned* rows,
                      unsigned n,
                      const float* queries,
                      int nq,
                      int dims,
                      float* out)
{
  score_mma_fp8<NT, false>(
    data, row_scales, row_bytes, blocks, n_codes, rows, n, queries, nq, dims, out, 0, 0, 0, 0);
}
/* The listed rows' squared distances at or under each query's bound, appended to its list. */
template <int NT>
RAFT_KERNEL fuse_fp8(const unsigned char* data,
                     const float* row_scales,
                     unsigned row_bytes,
                     unsigned blocks,
                     unsigned n_codes,
                     const unsigned* rows,
                     unsigned n,
                     const float* queries,
                     int nq,
                     int dims,
                     const float* bounds,
                     float* cand_d,
                     unsigned* cand_r,
                     unsigned* counts,
                     unsigned cap)
{
  score_mma_fp8<NT, true>(data,
                          row_scales,
                          row_bytes,
                          blocks,
                          n_codes,
                          rows,
                          n,
                          queries,
                          nq,
                          dims,
                          cand_d,
                          bounds,
                          cand_r,
                          counts,
                          cap);
}
/* Every (n / s)-th row of a row list: a sample that does not follow the list's order. */
RAFT_KERNEL stride_rows(const unsigned* rows, unsigned n, unsigned s, unsigned* out)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s) out[i] = rows[(unsigned long long)i * n / s];
}

}  // namespace cuvs::neighbors::cagra::detail::fp8_scan
