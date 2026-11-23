/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Row-list, selection and rerank kernels of the filtered scan (fp8_scan.cu).
 *
 * expand_count, expand_scan and expand_write turn a bitset into the rows it passes, in order.
 * bound_cut and prune_select_s take each query's sampled cutoff and its best candidates (a radix
 * select over keys in shared memory). rerank_h_i64, rerank8_h_i64 and rerank_final_h re-score the
 * candidates from FP16 rows with FP32 queries and double accumulation, and write the Euclidean
 * distance (the square root); the scan's output squares it. final_bucket, final_rank and
 * rerank_final_h keep each query's k best in order.
 */

#include <raft/core/detail/macros.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::fp8_scan {

__device__ unsigned int expand_word(const unsigned int* words,
                                    unsigned long long rows,
                                    unsigned long long w)
{
  unsigned long long nwords = (rows + 31) / 32;
  unsigned int bits         = w < nwords ? words[w] : 0u;
  if (w == nwords - 1 && rows % 32) bits &= (1u << (rows % 32)) - 1u;
  return bits;
}
RAFT_KERNEL expand_count(const unsigned int* words, unsigned long long rows, unsigned int* blocks)
{
  __shared__ unsigned int total;
  if (threadIdx.x == 0) total = 0;
  __syncthreads();
  unsigned int c =
    __popc(expand_word(words, rows, (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x));
  for (int o = 16; o > 0; o >>= 1)
    c += __shfl_down_sync(0xffffffffu, c, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(&total, c);
  __syncthreads();
  if (threadIdx.x == 0) blocks[blockIdx.x] = total;
}
RAFT_KERNEL expand_scan(unsigned int* blocks, unsigned int n)
{
  __shared__ unsigned int part[1024];
  unsigned int per = (n + 1023) / 1024, begin = threadIdx.x * per;
  unsigned int end = begin + per < n ? begin + per : n, sum = 0, run;
  for (unsigned int i = begin; i < end; i++)
    sum += blocks[i];
  part[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    run = 0;
    for (int t = 0; t < 1024; t++) {
      unsigned int v = part[t];
      part[t]        = run;
      run += v;
    }
  }
  __syncthreads();
  run = part[threadIdx.x];
  for (unsigned int i = begin; i < end; i++) {
    unsigned int v = blocks[i];
    blocks[i]      = run;
    run += v;
  }
}
RAFT_KERNEL expand_write(const unsigned int* words,
                         unsigned long long rows,
                         const unsigned int* blocks,
                         unsigned int* out)
{
  __shared__ unsigned int warps[32];
  unsigned long long w = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int bits = expand_word(words, rows, w), c = __popc(bits), x = c;
  unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 1; o < 32; o <<= 1) {
    unsigned int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) warps[warp] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int run = 0;
    for (unsigned int i = 0; i < blockDim.x / 32; i++) {
      unsigned int v = warps[i];
      warps[i]       = run;
      run += v;
    }
  }
  __syncthreads();
  unsigned int at = blocks[blockIdx.x] + warps[warp] + x - c;
  while (bits) {
    out[at++] = (unsigned int)(w * 32 + __ffs(bits) - 1);
    bits &= bits - 1;
  }
}

__device__ __forceinline__ unsigned warp_umin(unsigned v)
{
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(0xffffffffu, v);
#else
  for (int o = 16; o; o >>= 1)
    v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
#endif
}
__device__ __forceinline__ unsigned warp_umax(unsigned v)
{
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xffffffffu, v);
#else
  for (int o = 16; o; o >>= 1)
    v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
#endif
}
__device__ __forceinline__ float load(const unsigned short* p)
{
  float f;
  unsigned short h = *p;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

template <typename D, typename T>
__device__ void rerank(const D* data,
                       long long stride,
                       const unsigned* rows,
                       unsigned eligible,
                       unsigned nr,
                       T* positions,
                       unsigned top,
                       unsigned nq,
                       const float* queries,
                       int dims,
                       float* out)
{
  unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane   = threadIdx.x & 31;
  if (i >= top * nq) return;
  long long p  = (long long)positions[i];
  unsigned row = p < 0 || (unsigned long long)p >= eligible ? 0xffffffffu
                 : rows                                     ? rows[p]
                                                            : (unsigned)p;
  if (row >= nr) {
    if (!lane) {
      positions[i] = (T)-1;
      out[i]       = __int_as_float(0x7f800000);
    }
    return;
  }
  const D* v     = data + (long long)row * stride;
  const float* q = queries + (i / top) * dims;
  double sum     = 0.;
  for (int d = lane; d < dims; d += 32) {
    double x = (double)load(v + d) - q[d];
    sum += x * x;
  }
  for (int o = 16; o; o >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (!lane) {
    positions[i] = (T)row;
    out[i]       = (float)sqrt(sum);
  }
}
RAFT_KERNEL rerank_h_i64(const unsigned short* v,
                         long long st,
                         const unsigned* r,
                         unsigned ne,
                         unsigned nr,
                         long long* p,
                         unsigned top,
                         unsigned nq,
                         const float* q,
                         int d,
                         float* o)
{
  rerank(v, st, r, ne, nr, p, top, nq, q, d, o);
}

__device__ unsigned select_key(float f)
{
  unsigned b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : b | 0x80000000u;
}

RAFT_KERNEL bound_cut(const float* bounds, unsigned rank, unsigned q, float* out, unsigned* counts)
{
  __shared__ float part[32];
  unsigned row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* b = bounds + (unsigned long long)row * rank;
  float m        = -__int_as_float(0x7f800000);
  if (row >= q) return;
  for (unsigned i = tid; i < rank; i += blockDim.x)
    m = fmaxf(m, b[i]);
  for (int o = 16; o; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (lane == 0) part[warp] = m;
  __syncthreads();
  if (warp == 0) {
    m = lane < (blockDim.x >> 5) ? part[lane] : -__int_as_float(0x7f800000);
    for (int o = 16; o; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) {
      out[row * 32 + 31] = m;
      counts[row]        = 0;
    }
  }
}

__device__ __forceinline__ float select_float(unsigned u)
{
  return __uint_as_float((u & 0x80000000u) ? u & 0x7fffffffu : ~u);
}

/* One block a query: each candidate's rank by counting smaller keys. Dynamic shared memory: n
 * rounded up to 4 keys, then as many tie counters (2 * ((n + 3) & ~3) words). */
__launch_bounds__(1024) RAFT_KERNEL final_rank(
  const float* d, const long long* r, unsigned n, unsigned k, float* out_d, long long* out_r)
{
  extern __shared__ unsigned skey[];
  unsigned n4 = (n + 3) & ~3u, row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  unsigned* tie      = skey + n4;
  const float* v     = d + (unsigned long long)row * n;
  const long long* p = r + (unsigned long long)row * n;
  for (unsigned i = tid; i < n4; i += nt) {
    skey[i] = i < n ? select_key(v[i]) : 0xffffffffu;
    tie[i]  = 0;
  }
  for (unsigned i = n + tid; i < k; i += nt) {
    out_d[(unsigned long long)row * k + i] = __int_as_float(0x7f800000);
    out_r[(unsigned long long)row * k + i] = -1;
  }
  __syncthreads();
  const uint4* s4 = (const uint4*)skey;
  for (unsigned i = tid; i < n; i += nt) {
    unsigned ki = skey[i], c = 0;
    for (unsigned j = 0; j < n4 / 4; j++) {
      uint4 x = s4[j];
      c += (x.x < ki) + (x.y < ki) + (x.z < ki) + (x.w < ki);
    }
    c += atomicAdd(&tie[c], 1u);
    if (c < k) {
      out_d[(unsigned long long)row * k + c] = v[i];
      out_r[(unsigned long long)row * k + c] = p[i];
    }
  }
}

/* The query's next m (<= 8) coordinates: vector loads when aligned and a full step, else scalar. */
__device__ __forceinline__ void ldq8(const float* q, int m, float* qq)
{
  if (m == 8 && (((size_t)q) & 15) == 0) {
    float4 a = *(const float4*)q, b = *(const float4*)(q + 4);
    qq[0] = a.x;
    qq[1] = a.y;
    qq[2] = a.z;
    qq[3] = a.w;
    qq[4] = b.x;
    qq[5] = b.y;
    qq[6] = b.z;
    qq[7] = b.w;
  } else {
#pragma unroll
    for (int e = 0; e < 8; e++)
      qq[e] = e < m ? q[e] : 0.f;
  }
}
__device__ __forceinline__ double sq8(const unsigned short* v, const float* q, int m)
{
  uint4 x       = *(const uint4*)v;
  unsigned w[4] = {x.x, x.y, x.z, x.w};
  float qq[8];
  ldq8(q, m, qq);
  double s = 0.;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    float f;
    unsigned short h = (unsigned short)(w[e >> 1] >> ((e & 1) * 16));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    double t = (double)f - qq[e];
    s += e < m ? t * t : 0.;
  }
  return s;
}
/* Eight threads a row, eight dimensions a step (rows padded to a multiple of 8, 16-byte aligned).
 */
template <typename D>
__device__ void rerank8(const D* data,
                        long long stride,
                        const unsigned* rows,
                        unsigned eligible,
                        unsigned nr,
                        long long* positions,
                        unsigned top,
                        unsigned nq,
                        const float* queries,
                        int dims,
                        float* out)
{
  unsigned g    = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
  int sub       = threadIdx.x & 7;
  bool live     = g < top * nq;
  long long pos = live ? positions[g] : -1;
  unsigned row  = pos < 0 || (unsigned long long)pos >= eligible ? 0xffffffffu
                  : rows                                         ? rows[pos]
                                                                 : (unsigned)pos;
  bool ok       = live && row < nr;
  double sum    = 0.;
  if (ok) {
    const D* v     = data + (long long)row * stride;
    const float* q = queries + (unsigned long long)(g / top) * dims;
    for (int e = sub * 8; e < dims; e += 64)
      sum += sq8(v + e, q + e, dims - e < 8 ? dims - e : 8);
  }
  for (int o = 4; o; o >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (live && sub == 0) {
    positions[g] = ok ? (long long)row : -1;
    out[g]       = ok ? (float)sqrt(sum) : __int_as_float(0x7f800000);
  }
}
RAFT_KERNEL rerank8_h_i64(const unsigned short* v,
                          long long st,
                          const unsigned* r,
                          unsigned ne,
                          unsigned nr,
                          long long* p,
                          unsigned top,
                          unsigned nq,
                          const float* q,
                          int d,
                          float* o)
{
  rerank8(v, st, r, ne, nr, p, top, nq, q, d, o);
}

/* One block a query: the k smallest of its min(counts[q], cap) scores (cap without counts), as
 * (score, r[i] or i), by radix select. Dynamic shared memory: cap keys. */
__launch_bounds__(1024) RAFT_KERNEL prune_select_s(const float* d,
                                                   const unsigned* r,
                                                   const unsigned* counts,
                                                   unsigned cap,
                                                   unsigned k,
                                                   float* out_d,
                                                   long long* out_r)
{
  extern __shared__ unsigned keys[];
  __shared__ unsigned hist[2048], wsum[32], bu[256], bi[256], bp[256], sel[4096];
  __shared__ unsigned s_lo, s_hi, s_prefix, s_want, s_m, below, ties, nb;
  unsigned row = blockIdx.x, n = !counts || counts[row] > cap ? cap : (unsigned)counts[row];
  unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nt = blockDim.x;
  const float* v    = d + (unsigned long long)row * cap;
  const unsigned* p = r ? r + (unsigned long long)row * cap : 0;
  float* od         = out_d + (unsigned long long)row * k;
  long long* orow   = out_r + (unsigned long long)row * k;
  if (n <= k) {
    for (unsigned i = tid; i < k; i += nt) {
      od[i]   = i < n ? v[i] : __int_as_float(0x7f800000);
      orow[i] = i < n ? (long long)(p ? p[i] : i) : -1;
    }
    return;
  }
  unsigned lo = 0xffffffffu, hi = 0;
  for (unsigned i = tid; i < n; i += nt) {
    unsigned x = select_key(v[i]);
    keys[i]    = x;
    lo         = min(lo, x);
    hi         = max(hi, x);
  }
  if (tid == 0) {
    s_lo  = 0xffffffffu;
    s_hi  = 0;
    below = 0;
    ties  = 0;
    nb    = 0;
  }
  __syncthreads();
  lo = warp_umin(lo);
  hi = warp_umax(hi);
  if (lane == 0) {
    atomicMin(&s_lo, lo);
    atomicMax(&s_hi, hi);
  }
  __syncthreads();
  lo            = s_lo;
  hi            = s_hi;
  unsigned diff = lo ^ hi, want = k, prefix, mask, m = n;
  int top;
  if (diff == 0) {
    prefix = lo;
    mask   = 0xffffffffu;
    top    = -1;
  } else {
    top    = 31 - __clz(diff);
    mask   = top == 31 ? 0u : 0xffffffffu << (top + 1);
    prefix = lo & mask;
  }
  while (top >= 0 && m > 256) {
    int w = top + 1 < 11 ? top + 1 : 11, shift = top - w + 1;
    unsigned bins    = 1u << w;
    hist[tid]        = 0;
    hist[tid + 1024] = 0;
    __syncthreads();
    for (unsigned i = tid; i < n; i += nt) {
      unsigned x = keys[i];
      if ((x & mask) == prefix) atomicAdd(&hist[(x >> shift) & (bins - 1)], 1u);
    }
    __syncthreads();
    unsigned h0 = hist[2 * tid], h1 = hist[2 * tid + 1], c = h0 + h1;
    for (int o = 1; o < 32; o <<= 1) {
      unsigned y = __shfl_up_sync(0xffffffffu, c, o);
      if (lane >= o) c += y;
    }
    if (lane == 31) wsum[warp] = c;
    __syncthreads();
    if (warp == 0) {
      unsigned x = wsum[lane];
      for (int o = 1; o < 32; o <<= 1) {
        unsigned y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      wsum[lane] = x;
    }
    __syncthreads();
    unsigned e0 = (warp ? wsum[warp - 1] : 0) + c - h0 - h1;
    if (e0 < want && want <= e0 + h0) {
      s_prefix = prefix | ((2 * tid) << shift);
      s_want   = want - e0;
      s_m      = h0;
    } else if (e0 + h0 < want && want <= e0 + h0 + h1) {
      s_prefix = prefix | ((2 * tid + 1) << shift);
      s_want   = want - e0 - h0;
      s_m      = h1;
    }
    __syncthreads();
    prefix = s_prefix;
    want   = s_want;
    m      = s_m;
    mask |= (bins - 1) << shift;
    top = shift - 1;
  }

  bool small = m <= 256 && m != want;
  for (unsigned b0 = 0; b0 < n; b0 += nt) {
    unsigned i = b0 + tid, x = i < n ? keys[i] : 0xffffffffu, mk = x & mask;
    bool b       = i < n && mk < prefix;
    unsigned bal = __ballot_sync(0xffffffffu, b), base = 0;
    if (lane == 0 && bal) base = atomicAdd(&below, __popc(bal));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (b) {
      unsigned s = base + __popc(bal & ((1u << lane) - 1));
      od[s]      = select_float(x);
      sel[s]     = i;
    }
    if (i < n && mk == prefix) {
      if (small) {
        unsigned s = atomicAdd(&nb, 1u);
        bu[s]      = x;
        bi[s]      = i;
        bp[s]      = p ? p[i] : i;
      } else {
        unsigned t = atomicAdd(&ties, 1u);
        if (t < want) {
          unsigned s = k - want + t;
          od[s]      = select_float(x);
          sel[s]     = i;
        }
      }
    }
  }
  __syncthreads();
  if (small && tid < m) {
    unsigned uk = bu[tid], ik = bi[tid], pk = bp[tid], rk = 0;
    for (unsigned j = 0; j < m; j++) {
      unsigned x = bu[j];
      rk += x < uk || (x == uk && (bp[j] < pk || (bp[j] == pk && bi[j] < ik)));
    }
    if (rk < want) {
      unsigned s = k - want + rk;
      od[s]      = select_float(uk);
      sel[s]     = ik;
    }
  }
  __syncthreads();
  for (unsigned i = tid; i < k; i += nt)
    orow[i] = (long long)(p ? p[sel[i]] : sel[i]);
}

/* Thread tid < n holds candidate tid (f, pv); one block a query (blockIdx.x), 1024 threads. */
__device__ __forceinline__ void bucket_topk(
  float f, long long pv, unsigned n, unsigned k, float* out_d, long long* out_r)
{
  __shared__ unsigned hist[2048], start[2048], skey[1024], sidx[1024], wsum[32], s_lo, s_hi;
  unsigned row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  bool live = tid < n;
  for (unsigned i = n + tid; i < k; i += blockDim.x) {
    out_d[(unsigned long long)row * k + i] = __int_as_float(0x7f800000);
    out_r[(unsigned long long)row * k + i] = -1;
  }
  unsigned key = select_key(f), lo = live ? key : 0xffffffffu, hi = live ? key : 0u;
  hist[2 * tid]     = 0;
  hist[2 * tid + 1] = 0;
  if (tid == 0) {
    s_lo = 0xffffffffu;
    s_hi = 0;
  }
  __syncthreads();
  lo = warp_umin(lo);
  hi = warp_umax(hi);
  if (lane == 0) {
    atomicMin(&s_lo, lo);
    atomicMax(&s_hi, hi);
  }
  __syncthreads();
  lo             = s_lo;
  hi             = s_hi;
  unsigned range = hi - lo, shift = range < 2048 ? 0 : 21 - __clz(range);
  unsigned b = live ? (key - lo) >> shift : 0, slot = live ? atomicAdd(&hist[b], 1u) : 0;
  __syncthreads();
  unsigned h0 = hist[2 * tid], h1 = hist[2 * tid + 1], c = h0 + h1;
  for (int o = 1; o < 32; o <<= 1) {
    unsigned y = __shfl_up_sync(0xffffffffu, c, o);
    if (lane >= o) c += y;
  }
  if (lane == 31) wsum[warp] = c;
  __syncthreads();
  if (warp == 0) {
    unsigned x = lane < (blockDim.x >> 5) ? wsum[lane] : 0;
    for (int o = 1; o < 32; o <<= 1) {
      unsigned y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    wsum[lane] = x;
  }
  __syncthreads();
  unsigned e0        = (warp ? wsum[warp - 1] : 0) + c - h0 - h1;
  start[2 * tid]     = e0;
  start[2 * tid + 1] = e0 + h0;
  __syncthreads();
  unsigned base = start[b], cnt = hist[b];
  if (live) {
    skey[base + slot] = key;
    sidx[base + slot] = tid;
  }
  __syncthreads();
  if (!live) return;
  unsigned rank = base;
  for (unsigned j = base; j < base + cnt; j++) {
    unsigned x = skey[j];
    rank += x < key || (x == key && sidx[j] < tid);
  }
  if (rank < k) {
    out_d[(unsigned long long)row * k + rank] = f;
    out_r[(unsigned long long)row * k + rank] = pv;
  }
}
__launch_bounds__(1024) RAFT_KERNEL final_bucket(
  const float* d, const long long* r, unsigned n, unsigned k, float* out_d, long long* out_r)
{
  bool live             = threadIdx.x < n;
  unsigned long long at = (unsigned long long)blockIdx.x * n + threadIdx.x;
  bucket_topk(live ? d[at] : 0.f, live ? r[at] : -1, n, k, out_d, out_r);
}

__device__ __forceinline__ void ld8(const unsigned short* v, uint4& a, uint4& b)
{
  a = *(const uint4*)v;
}
__device__ __forceinline__ double sq8r(const unsigned short*, uint4 x, uint4, const float* q, int m)
{
  unsigned w[4] = {x.x, x.y, x.z, x.w};
  float qq[8];
  ldq8(q, m, qq);
  double s = 0.;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    float f;
    unsigned short h = (unsigned short)(w[e >> 1] >> ((e & 1) * 16));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    double t = (double)f - qq[e];
    s += e < m ? t * t : 0.;
  }
  return s;
}
template <typename D>
__device__ __forceinline__ void rerank_final(const D* data,
                                             long long stride,
                                             const unsigned* rows,
                                             unsigned eligible,
                                             unsigned nr,
                                             const long long* positions,
                                             unsigned n,
                                             unsigned k,
                                             const float* queries,
                                             int dims,
                                             float* out_d,
                                             long long* out_r)
{
  __shared__ float sf[1024];
  __shared__ long long sp[1024];
  unsigned row = blockIdx.x, tid = threadIdx.x, sub = tid & 3, per = blockDim.x >> 2;
  const float* qv     = queries + (unsigned long long)row * dims;
  const long long* pp = positions + (unsigned long long)row * n;
  long long pos[4];
  unsigned rr[4];
#pragma unroll
  for (int t = 0; t < 4; t++) {
    unsigned g = t * per + (tid >> 2);
    pos[t]     = g < n ? pp[g] : -1;
  }
#pragma unroll
  for (int t = 0; t < 4; t++)
    rr[t] = pos[t] < 0 || (unsigned long long)pos[t] >= eligible ? 0xffffffffu
            : rows                                               ? rows[pos[t]]
                                                                 : (unsigned)pos[t];
#pragma unroll 1
  for (int t = 0; t < 4; t++) {
    unsigned g = t * per + (tid >> 2);
    if (t * per >= n) break;
    bool ok    = g < n && rr[t] < nr;
    double sum = 0.;
    const D* v = data + (long long)(ok ? rr[t] : 0) * stride;
    for (int e0 = 0; e0 < dims; e0 += 128) {
      uint4 xa[4], xb[4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        int e = e0 + c * 32 + sub * 8;
        if (ok && e < dims) ld8(v + e, xa[c], xb[c]);
      }
#pragma unroll
      for (int c = 0; c < 4; c++) {
        int e = e0 + c * 32 + sub * 8;
        if (ok && e < dims) sum += sq8r(v, xa[c], xb[c], qv + e, dims - e < 8 ? dims - e : 8);
      }
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    if (g < n && sub == 0) {
      sf[g] = ok ? (float)sqrt(sum) : __int_as_float(0x7f800000);
      sp[g] = ok ? (long long)rr[t] : -1;
    }
  }
  __syncthreads();
  bool live = tid < n;
  bucket_topk(live ? sf[tid] : 0.f, live ? sp[tid] : -1, n, k, out_d, out_r);
}
__launch_bounds__(1024) RAFT_KERNEL rerank_final_h(const unsigned short* v,
                                                   long long st,
                                                   const unsigned* r,
                                                   unsigned ne,
                                                   unsigned nr,
                                                   const long long* p,
                                                   unsigned n,
                                                   unsigned k,
                                                   const float* q,
                                                   int d,
                                                   float* out_d,
                                                   long long* out_r)
{
  rerank_final(v, st, r, ne, nr, p, n, k, q, d, out_d, out_r);
}

}  // namespace cuvs::neighbors::cagra::detail::fp8_scan
