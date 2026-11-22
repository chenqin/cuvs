/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/postings.hpp>

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cuvs::core::postings {

namespace {

constexpr int kBlock         = 256;
constexpr int64_t kMaxBlocks = 16384;
constexpr uint32_t kAllBits  = 0xffffffffu;
constexpr int kOpAnd         = 0;
constexpr int kOpOr          = 1;
constexpr int kOpNot         = 2;
constexpr int kOpAll         = 3;
constexpr int kOpNone        = 4;
constexpr int kOpAndNot      = 5;  // a & ~b

/**
 * Set the bit of every row of consecutive runs that lies in [first_row, first_row + n_rows):
 * output position i is row `starts[r][i - (prefix[r] - prefix[0])]` of the run r whose prefix
 * (the rows of its leaf's runs before it) less the first run's is the largest at most i. Any
 * range of a leaf's runs, such as one run, can be scattered this way.
 */
__global__ void scatter_runs(const uint32_t* const* starts,
                             const int64_t* prefix,
                             int64_t n_runs,
                             int64_t total,
                             int64_t first_row,
                             int64_t n_rows,
                             uint32_t* words)
{
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t base = prefix[0];
    int64_t lo = 0, hi = n_runs;
    while (hi - lo > 1) {
      int64_t mid = (lo + hi) / 2;
      if (prefix[mid] - base <= i) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    int64_t row = static_cast<int64_t>(starts[lo][i - (prefix[lo] - base)]) - first_row;
    if (row >= 0 && row < n_rows) { atomicOr(&words[row >> 5], 1u << (row & 31)); }
  }
}

/** a = op(a, b) word by word, the last word cut to the bitset's rows. */
__global__ void combine_words(
  uint32_t* a, const uint32_t* b, int64_t n_words, int op, uint32_t tail)
{
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_words;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint32_t r;
    switch (op) {
      case kOpAnd: r = a[i] & b[i]; break;
      case kOpOr: r = a[i] | b[i]; break;
      case kOpNot: r = ~a[i]; break;
      case kOpAll: r = kAllBits; break;
      case kOpNone: r = 0u; break;
      default: r = a[i] & ~b[i]; break;
    }
    a[i] = i + 1 == n_words ? r & tail : r;
  }
}

/** Add the set bits of the words to *count. */
__global__ void count_bits(const uint32_t* words, int64_t n_words, unsigned long long* count)
{
  unsigned long long n = 0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_words;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    n += __popc(words[i]);
  }
  for (int d = 16; d > 0; d >>= 1) {
    n += __shfl_down_sync(kAllBits, n, d);
  }
  if ((threadIdx.x & 31) == 0 && n) { atomicAdd(count, n); }
}

auto blocks_for(int64_t n) -> unsigned
{
  return static_cast<unsigned>(std::clamp<int64_t>((n + kBlock - 1) / kBlock, 1, kMaxBlocks));
}

/** The most results the program holds at once; fails, saying why, on a malformed program. */
auto check_program(raft::host_vector_view<const step, int64_t> program, int64_t n_runs) -> int64_t
{
  int64_t depth = 0, deepest = 0;
  for (int64_t i = 0; i < program.extent(0); i++) {
    const step& s = program(i);
    switch (s.kind) {
      case step_kind::LEAF:
        RAFT_EXPECTS(s.first_run >= 0 && s.n_runs >= 0 && s.first_run + s.n_runs <= n_runs,
                     "postings::evaluate: step %ld names runs [%ld, %ld) of %ld",
                     static_cast<long>(i),
                     static_cast<long>(s.first_run),
                     static_cast<long>(s.first_run + s.n_runs),
                     static_cast<long>(n_runs));
        RAFT_EXPECTS(s.exclude_run < n_runs,
                     "postings::evaluate: step %ld excludes run %ld of %ld",
                     static_cast<long>(i),
                     static_cast<long>(s.exclude_run),
                     static_cast<long>(n_runs));
        [[fallthrough]];
      case step_kind::ALL:
      case step_kind::NONE: depth++; break;
      case step_kind::AND:
      case step_kind::OR:
        RAFT_EXPECTS(depth >= 2,
                     "postings::evaluate: step %ld combines two results of %ld",
                     static_cast<long>(i),
                     static_cast<long>(depth));
        depth--;
        break;
      case step_kind::NOT:
        RAFT_EXPECTS(
          depth >= 1, "postings::evaluate: step %ld negates no result", static_cast<long>(i));
        break;
      default:
        RAFT_FAIL("postings::evaluate: step %ld has an unknown kind %d",
                  static_cast<long>(i),
                  static_cast<int>(s.kind));
    }
    deepest = std::max(deepest, depth);
  }
  RAFT_EXPECTS(depth == 1,
               "postings::evaluate: the program leaves %ld results, not 1",
               static_cast<long>(depth));
  return deepest;
}

/** Words of the scratch bitsets: one for each result past the first, and one for an exclusion. */
auto scratch_words(int64_t deepest, int64_t n_rows) -> int64_t
{
  int64_t words = deepest * ((n_rows + 31) / 32);
  return words + (words & 1);  // keep the runs that follow 8-byte aligned
}

}  // namespace

auto workspace_size(raft::host_vector_view<const step, int64_t> program,
                    int64_t n_runs,
                    int64_t n_rows) -> int64_t
{
  RAFT_EXPECTS(n_rows >= 0 && n_runs >= 0, "postings::workspace_size: negative size");
  return scratch_words(check_program(program, n_runs), n_rows) + 4 * std::max<int64_t>(n_runs, 1);
}

void evaluate(
  raft::resources const& res,
  raft::host_vector_view<const step, int64_t> program,
  raft::host_vector_view<const run, int64_t> runs,
  raft::host_vector_view<const raft::device_vector_view<const uint32_t, int64_t>, int64_t> lists,
  cuvs::core::bitset_view<uint32_t, int64_t> out,
  int64_t first_row,
  std::optional<raft::device_vector_view<uint32_t, int64_t>> workspace,
  std::optional<raft::device_scalar_view<int64_t>> count)
{
  auto stream_ref      = raft::resource::get_cuda_stream(res);
  cudaStream_t stream  = stream_ref.get();
  const int64_t n_rows = out.size();
  const int64_t n_runs = runs.extent(0);
  const int64_t words  = (n_rows + 31) / 32;
  const uint32_t tail  = n_rows % 32 ? (1u << (n_rows % 32)) - 1 : kAllBits;

  RAFT_EXPECTS(first_row >= 0, "postings::evaluate: negative first_row");
  RAFT_EXPECTS(out.data() != nullptr || n_rows == 0, "postings::evaluate: null bitset");
  const int64_t deepest = check_program(program, n_runs);
  for (int64_t r = 0; r < n_runs; r++) {
    const run& x = runs(r);
    RAFT_EXPECTS(x.list < lists.extent(0),
                 "postings::evaluate: run %ld is in list %u of %ld",
                 static_cast<long>(r),
                 x.list,
                 static_cast<long>(lists.extent(0)));
    RAFT_EXPECTS(x.first >= 0 && x.count >= 0 && x.first + x.count <= lists(x.list).extent(0),
                 "postings::evaluate: run %ld covers [%ld, %ld) of a list of %ld rows",
                 static_cast<long>(r),
                 static_cast<long>(x.first),
                 static_cast<long>(x.first + x.count),
                 static_cast<long>(lists(x.list).extent(0)));
  }
  if (n_rows == 0) {
    if (count) { RAFT_CUDA_TRY(cudaMemsetAsync(count->data_handle(), 0, sizeof(int64_t), stream)); }
    return;
  }

  // Scratch: the result slots past out, then each run's start and its leaf's prefix of rows.
  const int64_t needed = workspace_size(program, n_runs, n_rows);
  std::optional<rmm::device_uvector<uint32_t>> owned;
  uint32_t* scratch = nullptr;
  if (workspace) {
    RAFT_EXPECTS(workspace->extent(0) >= needed,
                 "postings::evaluate: a workspace of %ld words, %ld needed",
                 static_cast<long>(workspace->extent(0)),
                 static_cast<long>(needed));
    RAFT_EXPECTS(reinterpret_cast<uintptr_t>(workspace->data_handle()) % 8 == 0,
                 "postings::evaluate: the workspace must be 8-byte aligned");
    scratch = workspace->data_handle();
  } else {
    owned.emplace(needed, stream_ref, raft::resource::get_workspace_resource_ref(res));
    scratch = owned->data();
  }
  const int64_t slot_words = scratch_words(deepest, n_rows);
  auto slot                = [&](int64_t depth) {
    return depth == 0 ? out.data() : scratch + (depth - 1) * words;
  };
  auto* device_starts = reinterpret_cast<const uint32_t**>(scratch + slot_words);
  auto* device_prefix =
    reinterpret_cast<int64_t*>(scratch + slot_words + 2 * std::max<int64_t>(n_runs, 1));

  // Each leaf's prefix of rows starts from 0; totals are kept for the launches.
  std::vector<const uint32_t*> starts(std::max<int64_t>(n_runs, 1), nullptr);
  std::vector<int64_t> prefix(std::max<int64_t>(n_runs, 1), 0);
  for (int64_t r = 0; r < n_runs; r++) {
    starts[r] = lists(runs(r).list).data_handle() + runs(r).first;
  }
  std::vector<int64_t> totals(program.extent(0), 0);
  for (int64_t i = 0; i < program.extent(0); i++) {
    const step& s = program(i);
    if (s.kind != step_kind::LEAF) { continue; }
    for (int64_t r = s.first_run; r < s.first_run + s.n_runs; r++) {
      prefix[r] = totals[i];
      totals[i] += runs(r).count;
    }
  }
  if (n_runs > 0) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      device_starts, starts.data(), n_runs * sizeof(starts[0]), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      device_prefix, prefix.data(), n_runs * sizeof(prefix[0]), cudaMemcpyHostToDevice, stream));
  }

  auto combine = [&](uint32_t* a, const uint32_t* b, int op) {
    combine_words<<<blocks_for(words), kBlock, 0, stream>>>(a, b, words, op, tail);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };
  auto scatter = [&](uint32_t* dest, int64_t first, int64_t n, int64_t total) {
    RAFT_CUDA_TRY(cudaMemsetAsync(dest, 0, words * sizeof(uint32_t), stream));
    if (total == 0) { return; }
    scatter_runs<<<blocks_for(total), kBlock, 0, stream>>>(
      device_starts + first, device_prefix + first, n, total, first_row, n_rows, dest);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };

  int64_t depth = 0;
  for (int64_t i = 0; i < program.extent(0); i++) {
    const step& s = program(i);
    switch (s.kind) {
      case step_kind::LEAF: {
        uint32_t* a = slot(depth);
        scatter(a, s.first_run, s.n_runs, totals[i]);
        if (s.complement) { combine(a, nullptr, kOpNot); }
        if (s.exclude_run >= 0) {
          // The excluded run's rows, scattered into the slot above, then removed.
          uint32_t* b = slot(depth + 1);
          scatter(b, s.exclude_run, 1, runs(s.exclude_run).count);
          combine(a, b, kOpAndNot);
        }
        depth++;
        break;
      }
      case step_kind::ALL: combine(slot(depth++), nullptr, kOpAll); break;
      case step_kind::NONE: combine(slot(depth++), nullptr, kOpNone); break;
      case step_kind::AND:
        combine(slot(depth - 2), slot(depth - 1), kOpAnd);
        depth--;
        break;
      case step_kind::OR:
        combine(slot(depth - 2), slot(depth - 1), kOpOr);
        depth--;
        break;
      case step_kind::NOT: combine(slot(depth - 1), nullptr, kOpNot); break;
    }
  }
  if (count) {
    RAFT_CUDA_TRY(cudaMemsetAsync(count->data_handle(), 0, sizeof(int64_t), stream));
    count_bits<<<blocks_for(words), kBlock, 0, stream>>>(
      out.data(), words, reinterpret_cast<unsigned long long*>(count->data_handle()));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace cuvs::core::postings
