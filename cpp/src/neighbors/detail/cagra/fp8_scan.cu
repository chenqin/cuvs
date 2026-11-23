/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * The FP8 filtered scan of CAGRA search (search_params::filtered_route). fp8_scan.hpp describes
 * the steps; fp8_scan_kernel.cuh and fp8_scan_select_kernel.cuh hold the kernels.
 */

#include "../../../core/nvtx.hpp"
#include "fp8_scan.hpp"
#include "fp8_scan_kernel.cuh"
#include "fp8_scan_select_kernel.cuh"
#include "utils.hpp"

#include <cuvs/selection/select_k.hpp>

#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/custom_resource.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <vector>

namespace cuvs::neighbors::cagra::detail::fp8_scan {

namespace fp8 = cuvs::preprocessing::quantize::fp8;

constexpr unsigned kGroupMax  = 64;                 // queries one scan kernel takes
constexpr unsigned kSample    = 8192;               // sampled scores per query
constexpr unsigned kRankFloor = 32;                 // the sample's least cutoff rank
constexpr unsigned kMargin    = 4;                  // expected kept / needed candidates
constexpr unsigned kCap       = 32768;              // most scores kept per query
constexpr unsigned kPruneMin  = 65536;              // fewer rows select in full
constexpr unsigned kFinalMax  = 4096;               // most candidates final_rank ranks
constexpr size_t kMatrixBytes = size_t{512} << 20;  // score matrix of the full selection, at most

/*
 * What the graph search leaves in the slots past the rows it found: its invalid internal id
 * (utils::get_max_value<uint32_t>(), the fill in search_multi_cta_jit.cuh) widened to the output
 * type as search_multi_cta.cuh widens ids, and utils::get_max_value<float>().
 */
template <typename OutputIdxT>
inline constexpr OutputIdxT kUnfilledNeighbor =
  static_cast<OutputIdxT>(utils::get_max_value<uint32_t>());

RAFT_KERNEL queries_to_float(const half* in, int64_t n, float* out)
{
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = __half2float(in[i]); }
}

/*
 * Results [nq][k] from each query's kout found ones, with the distances squared; slots past them
 * get kUnfilledNeighbor and the largest float. With retry, each query's flag: 1 when its kept count
 * (counts, from the sampled cutoff) is outside [c, kCap], so its candidates were not the c best.
 */
template <typename OutputIdxT>
RAFT_KERNEL finish(const float* found_d,
                   const long long* found_r,
                   unsigned nq,
                   unsigned kout,
                   unsigned k,
                   OutputIdxT* neighbors,
                   int64_t ld_n,
                   float* distances,
                   int64_t ld_d,
                   const uint32_t* counts,
                   unsigned c,
                   uint32_t* retry)
{
  unsigned q = blockIdx.x;
  if (q >= nq) { return; }
  if (retry != nullptr && threadIdx.x == 0) {
    retry[q] = counts != nullptr && (counts[q] < c || counts[q] > kCap) ? 1u : 0u;
  }
  for (unsigned j = threadIdx.x; j < k; j += blockDim.x) {
    long long r             = j < kout ? found_r[static_cast<uint64_t>(q) * kout + j] : -1;
    float d                 = j < kout ? found_d[static_cast<uint64_t>(q) * kout + j] : 0.0f;
    bool ok                 = r >= 0 && isfinite(d);
    neighbors[q * ld_n + j] = ok ? static_cast<OutputIdxT>(r) : kUnfilledNeighbor<OutputIdxT>;
    distances[q * ld_d + j] = ok ? d * d : utils::get_max_value<float>();
  }
}

RAFT_KERNEL iota_rows(uint32_t* out, int64_t n)
{
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = static_cast<uint32_t>(i); }
}

namespace {

/* Per device, set up once: whether the scan runs there, and the limits of its kernels. */
struct device_setup {
  bool supported         = false;  // the device loads an sm_89 or newer image of the kernels
  unsigned scan_shared   = 0;      // dynamic shared memory of a scan kernel
  unsigned select_shared = 0;      // dynamic shared memory of prune_select_s
  unsigned sms           = 0;
};

struct device_slot {
  std::once_flag once;
  device_setup setup;
};

/*
 * One slot per device of the process, made on the first call (not at load, so loading the library
 * does not initialise CUDA) and never resized: after its setup, reading a slot takes no lock.
 */
auto device_slots() -> std::vector<device_slot>&
{
  static std::vector<device_slot> slots(static_cast<size_t>([] {
    int count = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
    return count;
  }()));
  return slots;
}

template <typename Kernel>
auto opt_in(Kernel kernel, unsigned shared_limit, unsigned want) -> unsigned
{
  cudaFuncAttributes attr{};
  RAFT_CUDA_TRY(cudaFuncGetAttributes(&attr, kernel));
  unsigned room    = shared_limit > attr.sharedSizeBytes
                       ? shared_limit - static_cast<unsigned>(attr.sharedSizeBytes)
                       : 0u;
  unsigned dynamic = std::min(want, room);
  RAFT_CUDA_TRY(cudaFuncSetAttribute(
    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic)));
  return dynamic;
}

/* Dynamic shared memory of final_rank for c candidates: c keys and c tie counters, each rounded
 * up to 4. */
constexpr auto final_rank_shared(unsigned c) -> unsigned
{
  return 2 * ((c + 3) & ~3u) * static_cast<unsigned>(sizeof(unsigned));
}

/*
 * The setup of the current device, which is the device of res as for every launch of the scan.
 * Whether the scan runs is the virtual architecture of the scan kernel image the driver loads
 * (cudaFuncAttributes::ptxVersion, as raft::util::arch::kernel_virtual_arch reads it): images
 * older than sm_89 hold empty scan kernels, and a device without a loadable image has none.
 */
auto make_setup(raft::resources const& res) -> device_setup
{
  device_setup s;
  cudaFuncAttributes attr{};
  cudaError_t image = cudaFuncGetAttributes(&attr, score_fp8<2>);
  if (image == cudaErrorInvalidDeviceFunction || image == cudaErrorNoKernelImageForDevice) {
    (void)cudaGetLastError();
    return s;
  }
  RAFT_CUDA_TRY(image);
  if (attr.ptxVersion < 89) { return s; }
  const auto& props       = raft::resource::get_device_properties(res);
  auto shared_limit       = static_cast<unsigned>(props.sharedMemPerBlockOptin);
  s.sms                   = static_cast<unsigned>(props.multiProcessorCount);
  constexpr unsigned want = 160 * 1024;
  s.scan_shared           = want;
  for (unsigned got : {opt_in(score_fp8<2>, shared_limit, want),
                       opt_in(score_fp8<4>, shared_limit, want),
                       opt_in(score_fp8<8>, shared_limit, want),
                       opt_in(fuse_fp8<2>, shared_limit, want),
                       opt_in(fuse_fp8<4>, shared_limit, want),
                       opt_in(fuse_fp8<8>, shared_limit, want)}) {
    s.scan_shared = std::min(s.scan_shared, got);
  }
  s.select_shared = opt_in(prune_select_s, shared_limit, std::max<unsigned>(kCap * 4, kSample * 4));
  (void)opt_in(final_rank, shared_limit, final_rank_shared(kFinalMax));
  s.supported = true;
  return s;
}

/* The device's setup, made by the first call on it. A failed setup throws and is tried again. */
auto setup(raft::resources const& res) -> const device_setup&
{
  auto& slots = device_slots();
  int device  = raft::resource::get_device_id(res);
  RAFT_EXPECTS(device >= 0 && static_cast<size_t>(device) < slots.size(),
               "CAGRA filtered scan: device %d out of range",
               device);
  auto& slot = slots[device];
  std::call_once(slot.once, [&] { slot.setup = make_setup(res); });
  return slot.setup;
}

/* Halves a query takes in a scan kernel's shared memory. */
auto query_stride(unsigned dims) -> unsigned { return ((dims + 15) & ~15u) + kQueryPad; }

/* Most queries a scan kernel takes at these dimensions (the fused scan's hit lists included). */
auto group_limit(const device_setup& s, unsigned dims) -> unsigned
{
  for (unsigned q = kGroupMax; q >= 16; q /= 2) {
    if (fp8_query_bytes(q, query_stride(dims)) + fp8_hit_bytes <= s.scan_shared) { return q; }
  }
  return 0;
}

/* 16-byte blocks of a row the scan reads: the last one left out when the caller says so. */
auto scan_blocks(unsigned row_bytes, unsigned dims, bool skip_last_block) -> unsigned
{
  unsigned blocks = std::min(row_bytes / 16, (dims + 15) / 16);
  if (skip_last_block && blocks > 1) { blocks--; }
  return blocks;
}

auto div_up(uint64_t a, uint64_t b) -> unsigned { return static_cast<unsigned>((a + b - 1) / b); }

/* Squared-distance matrix [nq][n] of the listed rows for nq <= group_limit queries. */
void score_matrix(const device_setup& s,
                  const fp8::quantizer_view& codes,
                  unsigned blocks,
                  const uint32_t* rows,
                  unsigned n,
                  const float* queries,
                  unsigned nq,
                  unsigned dims,
                  float* out,
                  cudaStream_t stream)
{
  unsigned grid    = std::min(div_up(n, 128), s.sms * 4);
  unsigned width   = nq <= 16 ? 0 : nq <= 32 ? 1 : 2;
  unsigned shared  = fp8_query_bytes(16u << width, query_stride(dims));
  auto data        = codes.codes.data_handle();
  auto scales      = codes.row_scales.data_handle();
  unsigned rb      = codes.encoded_row_length();
  unsigned n_codes = static_cast<unsigned>(codes.n_rows());
  int q = static_cast<int>(nq), d = static_cast<int>(dims);
  if (width == 0) {
    score_fp8<2><<<grid, kScanThreads, shared, stream>>>(
      data, scales, rb, blocks, n_codes, rows, n, queries, q, d, out);
  } else if (width == 1) {
    score_fp8<4><<<grid, kScanThreads, shared, stream>>>(
      data, scales, rb, blocks, n_codes, rows, n, queries, q, d, out);
  } else {
    score_fp8<8><<<grid, kScanThreads, shared, stream>>>(
      data, scales, rb, blocks, n_codes, rows, n, queries, q, d, out);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/* The listed rows' squared distances at or under each query's bound, appended to its list. */
void score_below(const device_setup& s,
                 const fp8::quantizer_view& codes,
                 unsigned blocks,
                 const uint32_t* rows,
                 unsigned n,
                 const float* queries,
                 unsigned nq,
                 unsigned dims,
                 const float* bounds,
                 float* cand_d,
                 uint32_t* cand_r,
                 uint32_t* counts,
                 cudaStream_t stream)
{
  unsigned grid    = std::min(div_up(n, 128), s.sms * 2);
  unsigned width   = nq <= 16 ? 0 : nq <= 32 ? 1 : 2;
  unsigned shared  = fp8_query_bytes(16u << width, query_stride(dims)) + fp8_hit_bytes;
  auto data        = codes.codes.data_handle();
  auto scales      = codes.row_scales.data_handle();
  unsigned rb      = codes.encoded_row_length();
  unsigned n_codes = static_cast<unsigned>(codes.n_rows());
  int q = static_cast<int>(nq), d = static_cast<int>(dims);
  if (width == 0) {
    fuse_fp8<2><<<grid, kScanThreads, shared, stream>>>(data,
                                                        scales,
                                                        rb,
                                                        blocks,
                                                        n_codes,
                                                        rows,
                                                        n,
                                                        queries,
                                                        q,
                                                        d,
                                                        bounds,
                                                        cand_d,
                                                        cand_r,
                                                        counts,
                                                        kCap);
  } else if (width == 1) {
    fuse_fp8<4><<<grid, kScanThreads, shared, stream>>>(data,
                                                        scales,
                                                        rb,
                                                        blocks,
                                                        n_codes,
                                                        rows,
                                                        n,
                                                        queries,
                                                        q,
                                                        d,
                                                        bounds,
                                                        cand_d,
                                                        cand_r,
                                                        counts,
                                                        kCap);
  } else {
    fuse_fp8<8><<<grid, kScanThreads, shared, stream>>>(data,
                                                        scales,
                                                        rb,
                                                        blocks,
                                                        n_codes,
                                                        rows,
                                                        n,
                                                        queries,
                                                        q,
                                                        d,
                                                        bounds,
                                                        cand_d,
                                                        cand_r,
                                                        counts,
                                                        kCap);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/* The cutoff rank of the sample for c candidates over n rows; 0 when the fused selection does not
 * apply. */
auto fused_rank(const device_setup& s, unsigned n, unsigned c, unsigned dims) -> unsigned
{
  if (n < kPruneMin || c > kCap / kMargin || dims > max_dims || s.select_shared < kCap * 4) {
    return 0;
  }
  uint64_t rank = (uint64_t{kMargin} * c * kSample + n - 1) / n;
  rank          = std::max<uint64_t>(rank, kRankFloor);
  if (rank > kSample / 4 || uint64_t{n} * rank / kSample > kCap / 2) { return 0; }
  return static_cast<unsigned>(rank);
}

/* Takes 256-byte aligned pieces of one workspace buffer; with no buffer, only counts the bytes. */
struct carve {
  char* base  = nullptr;
  size_t used = 0;
  template <typename T>
  auto take(size_t n) -> T*
  {
    T* out = base != nullptr ? reinterpret_cast<T*>(base + used) : nullptr;
    used += (n * sizeof(T) + 255) & ~size_t{255};
    return out;
  }
};

/* The device buffers of one group of nq queries. */
struct group_buffers {
  float* cand_d;
  long long* cand_p;
  uint32_t* sample_rows;
  float* sample;
  float* cut;
  long long* cut_rows;
  float* bounds;
  uint32_t* counts;
  float* kept_d;
  uint32_t* kept_r;
  float* out_d;
  long long* out_r;
};

auto take_group(carve& w, unsigned nq, unsigned c, unsigned kout, unsigned rank) -> group_buffers
{
  group_buffers b{};
  b.cand_d = w.take<float>(size_t{nq} * c);
  b.cand_p = w.take<long long>(size_t{nq} * c);
  if (rank > 0) {
    b.sample_rows = w.take<uint32_t>(kSample);
    b.sample      = w.take<float>(size_t{nq} * kSample);
    b.cut         = w.take<float>(size_t{nq} * rank);
    b.cut_rows    = w.take<long long>(size_t{nq} * rank);
    b.bounds      = w.take<float>(size_t{nq} * 32);
    b.counts      = w.take<uint32_t>(nq);
    b.kept_d      = w.take<float>(size_t{nq} * kCap);
    b.kept_r      = w.take<uint32_t>(size_t{nq} * kCap);
  }
  b.out_d = w.take<float>(size_t{nq} * kout);
  b.out_r = w.take<long long>(size_t{nq} * kout);
  return b;
}

/*
 * The host copy of a group's kept counts: pinned, made on first use and held by res (one search
 * at a time per raft::resources), so waiting for the counts allocates nothing.
 */
struct pinned_counts {
  std::optional<raft::pinned_vector<uint32_t, std::size_t>> buffer;
};

auto kept_counts_host(raft::resources const& res) -> uint32_t*
{
  auto* counts = raft::resource::get_custom_resource<pinned_counts>(res);
  if (!counts->buffer.has_value()) {
    counts->buffer.emplace(raft::make_pinned_vector<uint32_t, std::size_t>(res, kGroupMax));
  }
  return counts->buffer->data_handle();
}

/*
 * The workspace when bytes take at most half of what it has free, so that select_k and the
 * per-group buffers that follow still fit there; else the large workspace.
 */
auto buffer_resource(raft::resources const& res, size_t bytes) -> rmm::device_async_resource_ref
{
  if (bytes <= raft::resource::get_workspace_free_bytes(res) / 2) {
    return raft::resource::get_workspace_resource_ref(res);
  }
  return raft::resource::get_large_workspace_resource_ref(res);
}

/* The full selection's score matrix: made by the first group that needs it, kept for the others. */
struct score_buffer {
  std::optional<rmm::device_uvector<float>> scores;
  unsigned chunk = 0;  // queries it holds
};

/* Where a group's results are: each query's kout best (distance, row), and the kept counts of the
 * fused selection (null when the full selection ran). */
struct group_results {
  const float* out_d     = nullptr;
  const long long* out_r = nullptr;
  const uint32_t* counts = nullptr;
};

/* The queries of one group, nq <= group_limit, all over the same n > 0 listed rows. */
auto search_group(raft::resources const& res,
                  const device_setup& s,
                  const fp8::quantizer_view& codes,
                  unsigned blocks,
                  const rerank_rows& dataset,
                  const float* queries,
                  unsigned nq,
                  const uint32_t* rows,
                  unsigned n,
                  unsigned c,
                  unsigned kk,
                  unsigned rank,
                  bool flags,
                  char* workspace,
                  score_buffer& matrix) -> group_results
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  unsigned dims       = dataset.dim;

  // Candidates: their FP8 distances and positions in rows, [nq][c].
  unsigned kout = std::min(kk, c);
  carve w{workspace};
  group_buffers b   = take_group(w, nq, c, kout, rank);
  float* cand_d     = b.cand_d;
  long long* cand_p = b.cand_p;

  bool selected = false;
  if (rank > 0) {
    stride_rows<<<div_up(kSample, 256), 256, 0, stream>>>(rows, n, kSample, b.sample_rows);
    score_matrix(s, codes, blocks, b.sample_rows, kSample, queries, nq, dims, b.sample, stream);
    prune_select_s<<<nq, 1024, kSample * 4, stream>>>(
      b.sample, nullptr, nullptr, kSample, rank, b.cut, b.cut_rows);
    bound_cut<<<nq, 256, 0, stream>>>(b.cut, rank, nq, b.bounds, b.counts);
    score_below(
      s, codes, blocks, rows, n, queries, nq, dims, b.bounds, b.kept_d, b.kept_r, b.counts, stream);
    prune_select_s<<<nq, 1024, kCap * 4, stream>>>(
      b.kept_d, b.kept_r, b.counts, kCap, c, cand_d, cand_p);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Every query must have kept between c and the cap: the candidates under the cutoff are then
    // exactly the c best. With retry flags the caller checks, from the flags finish writes;
    // otherwise wait for the counts here.
    if (flags) {
      selected = true;
    } else {
      uint32_t* kept = kept_counts_host(res);
      raft::copy(kept, b.counts, nq, stream);
      raft::resource::sync_stream(res, stream);
      selected = std::all_of(kept, kept + nq, [&](uint32_t x) { return x >= c && x <= kCap; });
    }
  }

  if (!selected) {
    if (!matrix.scores.has_value()) {
      // As many queries a chunk as kMatrixBytes holds, at least one.
      uint64_t row_bytes = uint64_t{n} * sizeof(float);
      matrix.chunk = static_cast<unsigned>(std::clamp<uint64_t>(kMatrixBytes / row_bytes, 1, nq));
      size_t count = size_t{matrix.chunk} * n;
      matrix.scores.emplace(count, stream, buffer_resource(res, count * sizeof(float)));
    }
    float* scores = matrix.scores->data();
    for (unsigned q0 = 0; q0 < nq; q0 += matrix.chunk) {
      unsigned m = std::min(matrix.chunk, nq - q0);
      score_matrix(s, codes, blocks, rows, n, queries + size_t{q0} * dims, m, dims, scores, stream);
      cuvs::selection::select_k(
        res,
        raft::make_device_matrix_view<const float, int64_t>(scores, m, n),
        std::nullopt,
        raft::make_device_matrix_view<float, int64_t>(cand_d + size_t{q0} * c, m, c),
        raft::make_device_matrix_view<int64_t, int64_t>(
          reinterpret_cast<int64_t*>(cand_p) + size_t{q0} * c, m, c),
        true);
    }
  }

  // Re-score from FP16 rows and keep the k best.
  long long stride = dataset.stride;
  unsigned nr      = static_cast<unsigned>(dataset.n_rows);
  int d            = static_cast<int>(dims);
  auto data        = reinterpret_cast<const unsigned short*>(dataset.data);
  // The 16-byte loads of rerank8 and rerank_final: whole steps of 8 halves, aligned bases.
  bool wide = stride >= static_cast<long long>((dims + 7) & ~7u) && stride % 8 == 0 &&
              reinterpret_cast<uintptr_t>(dataset.data) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(queries) % 16 == 0;
  RAFT_EXPECTS(c <= kFinalMax, "CAGRA filtered scan: %u candidates, at most %u", c, kFinalMax);
  if (c <= 1024 && wide) {
    rerank_final_h<<<nq, 1024, 0, stream>>>(
      data, stride, rows, n, nr, cand_p, c, kout, queries, d, b.out_d, b.out_r);
  } else {
    if (wide) {
      rerank8_h_i64<<<div_up(uint64_t{nq} * c * 8, 256), 256, 0, stream>>>(
        data, stride, rows, n, nr, cand_p, c, nq, queries, d, cand_d);
    } else {
      rerank_h_i64<<<div_up(uint64_t{nq} * c * 32, 256), 256, 0, stream>>>(
        data, stride, rows, n, nr, cand_p, c, nq, queries, d, cand_d);
    }
    if (c <= 1024) {
      final_bucket<<<nq, 1024, 0, stream>>>(cand_d, cand_p, c, kout, b.out_d, b.out_r);
    } else {
      final_rank<<<nq, 1024, final_rank_shared(c), stream>>>(
        cand_d, cand_p, c, kout, b.out_d, b.out_r);
    }
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return group_results{b.out_d, b.out_r, rank > 0 && selected ? b.counts : nullptr};
}

}  // namespace

auto is_supported(raft::resources const& res) -> bool
{
  try {
    return setup(res).supported;
  } catch (const std::exception&) {
    // The setup could not run (RAFT_CUDA_TRY has cleared the CUDA error); it is tried again on the
    // next call.
    return false;
  }
}

template <typename OutputIdxT>
void search(raft::resources const& res,
            const cagra::search_params& params,
            const fp8::quantizer_view& codes,
            const rerank_rows& dataset,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            const uint32_t* rows,
            uint32_t n_rows,
            raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  int64_t nq    = queries.extent(0);
  int64_t kk    = neighbors.extent(1);
  unsigned dims = dataset.dim;
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::fp8_scan(n_queries = %ld, k = %ld, rows = %ld)",
    static_cast<long>(nq),
    static_cast<long>(kk),
    static_cast<long>(n_rows));
  RAFT_EXPECTS(codes.n_rows() == dataset.n_rows && dataset.n_rows < (int64_t{1} << 32),
               "CAGRA filtered scan: %ld code rows for %ld rows (fewer than 2^32 needed)",
               static_cast<long>(codes.n_rows()),
               static_cast<long>(dataset.n_rows));
  RAFT_EXPECTS(codes.n_rows() == 0 || (codes.codes.data_handle() != nullptr &&
                                       codes.row_scales.data_handle() != nullptr),
               "CAGRA filtered scan: the index has no scan codes");
  RAFT_EXPECTS(
    dims > 0 && dims <= max_dims, "CAGRA filtered scan takes 1 to %u dimensions", max_dims);
  RAFT_EXPECTS(
    codes.encoded_row_length() >= (dims + 15) / 16 * 16 && codes.encoded_row_length() % 16 == 0,
    "CAGRA filtered scan: scan code rows are too short for the dimension");
  RAFT_EXPECTS(
    kk > 0 && kk <= max_k, "CAGRA filtered scan takes k of 1 to %ld", static_cast<long>(max_k));
  RAFT_EXPECTS(queries.extent(1) == static_cast<int64_t>(dims),
               "CAGRA filtered scan: queries of %ld dimensions for rows of %u",
               static_cast<long>(queries.extent(1)),
               dims);
  RAFT_EXPECTS(neighbors.extent(0) == nq && distances.extent(0) == nq && distances.extent(1) == kk,
               "CAGRA filtered scan: neighbors and distances must be [queries, k]");
  RAFT_EXPECTS(rows != nullptr || n_rows == 0, "CAGRA filtered scan: no row list");
  if (nq == 0) { return; }

  const auto& s = setup(res);
  RAFT_EXPECTS(s.supported,
               "CAGRA filtered scan: the FP8 scan kernels are not compiled for this device (an "
               "sm_89 or newer image is needed)");
  unsigned limit = group_limit(s, dims);
  RAFT_EXPECTS(limit > 0, "CAGRA filtered scan: too little shared memory for %u dimensions", dims);

  // Candidates per query: scan_candidates, or k plus three times k (at least 16 more), within the
  // rows listed.
  uint64_t want   = params.scan_candidates > 0 ? std::max<uint64_t>(params.scan_candidates, kk)
                                               : kk + std::max<uint64_t>(16, 3 * kk);
  unsigned c      = static_cast<unsigned>(std::min<uint64_t>({want, n_rows, kFinalMax}));
  c               = std::max<unsigned>(c, std::min<uint64_t>(kk, n_rows));
  unsigned rank   = n_rows > 0 && !params.scan_full_selection ? fused_rank(s, n_rows, c, dims) : 0;
  unsigned kout   = std::min<unsigned>(static_cast<unsigned>(kk), c);
  unsigned blocks = scan_blocks(codes.encoded_row_length(), dims, params.scan_skip_last_block);
  uint32_t* retry = params.scan_retry;

  // One workspace allocation per call: the queries as floats, then one group's buffers, reused by
  // every group.
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  carve sizes;
  sizes.take<float>(static_cast<size_t>(nq) * dims);
  take_group(sizes, static_cast<unsigned>(std::min<int64_t>(limit, nq)), c, kout, rank);
  rmm::device_uvector<char> workspace(
    sizes.used, stream, raft::resource::get_workspace_resource_ref(res));
  carve w{workspace.data()};
  float* qf = w.take<float>(static_cast<size_t>(nq) * dims);
  queries_to_float<<<div_up(static_cast<uint64_t>(nq) * dims, 256), 256, 0, stream>>>(
    queries.data_handle(), nq * dims, qf);
  score_buffer matrix;
  for (int64_t q0 = 0; q0 < nq; q0 += limit) {
    unsigned m = static_cast<unsigned>(std::min<int64_t>(limit, nq - q0));
    // With no rows listed nothing passes, and every result is empty.
    group_results found;
    if (n_rows > 0) {
      found = search_group(res,
                           s,
                           codes,
                           blocks,
                           dataset,
                           qf + q0 * dims,
                           m,
                           rows,
                           n_rows,
                           c,
                           static_cast<unsigned>(kk),
                           rank,
                           retry != nullptr,
                           workspace.data() + w.used,
                           matrix);
    }
    finish<OutputIdxT><<<m, 256, 0, stream>>>(found.out_d,
                                              found.out_r,
                                              m,
                                              kout,
                                              static_cast<unsigned>(kk),
                                              neighbors.data_handle() + q0 * neighbors.extent(1),
                                              neighbors.extent(1),
                                              distances.data_handle() + q0 * distances.extent(1),
                                              distances.extent(1),
                                              found.counts,
                                              c,
                                              retry != nullptr ? retry + q0 : nullptr);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

template void search<uint32_t>(raft::resources const&,
                               const cagra::search_params&,
                               const fp8::quantizer_view&,
                               const rerank_rows&,
                               raft::device_matrix_view<const half, int64_t, raft::row_major>,
                               const uint32_t*,
                               uint32_t,
                               raft::device_matrix_view<uint32_t, int64_t, raft::row_major>,
                               raft::device_matrix_view<float, int64_t, raft::row_major>);
template void search<int64_t>(raft::resources const&,
                              const cagra::search_params&,
                              const fp8::quantizer_view&,
                              const rerank_rows&,
                              raft::device_matrix_view<const half, int64_t, raft::row_major>,
                              const uint32_t*,
                              uint32_t,
                              raft::device_matrix_view<int64_t, int64_t, raft::row_major>,
                              raft::device_matrix_view<float, int64_t, raft::row_major>);

auto bitset_rows(raft::resources const& res, const uint32_t* words, int64_t n_rows)
  -> rmm::device_uvector<uint32_t>
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto mr             = raft::resource::get_workspace_resource_ref(res);
  RAFT_EXPECTS(n_rows >= 0 && n_rows < (int64_t{1} << 32),
               "CAGRA filtered scan takes fewer than 2^32 rows");
  uint64_t nwords = (static_cast<uint64_t>(n_rows) + 31) / 32;
  unsigned grid   = std::max(div_up(nwords, 256), 1u);
  rmm::device_uvector<unsigned> block_counts(grid, stream, mr);
  auto rows = static_cast<unsigned long long>(n_rows);
  expand_count<<<grid, 256, 0, stream>>>(words, rows, block_counts.data());
  unsigned last_count = 0, last_start = 0;
  raft::copy(&last_count, block_counts.data() + grid - 1, 1, stream);
  expand_scan<<<1, 1024, 0, stream>>>(block_counts.data(), grid);
  raft::copy(&last_start, block_counts.data() + grid - 1, 1, stream);
  raft::resource::sync_stream(res, stream);
  size_t count = size_t{last_start} + last_count;
  rmm::device_uvector<uint32_t> out(count, stream, buffer_resource(res, count * sizeof(uint32_t)));
  if (out.size() > 0) {
    expand_write<<<grid, 256, 0, stream>>>(words, rows, block_counts.data(), out.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  return out;
}

auto all_rows(raft::resources const& res, int64_t n_rows) -> rmm::device_uvector<uint32_t>
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  RAFT_EXPECTS(n_rows >= 0 && n_rows < (int64_t{1} << 32),
               "CAGRA filtered scan takes fewer than 2^32 rows");
  rmm::device_uvector<uint32_t> out(
    n_rows, stream, buffer_resource(res, static_cast<size_t>(n_rows) * sizeof(uint32_t)));
  if (n_rows > 0) {
    iota_rows<<<div_up(n_rows, 256), 256, 0, stream>>>(out.data(), n_rows);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  return out;
}

}  // namespace cuvs::neighbors::cagra::detail::fp8_scan
