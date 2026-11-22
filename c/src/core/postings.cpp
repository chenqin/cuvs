/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <optional>
#include <vector>

#include <cuvs/core/c_api.h>
#include <cuvs/core/postings.h>
#include <cuvs/core/postings.hpp>

#include "exceptions.hpp"
#include "interop.hpp"

namespace {

namespace postings = cuvs::core::postings;

auto to_steps(const cuvsPostingsStep *steps, int64_t n_steps)
    -> std::vector<postings::step> {
  RAFT_EXPECTS(n_steps >= 0 && (steps != nullptr || n_steps == 0),
               "cuvsPostings: null steps");
  std::vector<postings::step> out(n_steps);
  for (int64_t i = 0; i < n_steps; i++) {
    out[i] = postings::step{static_cast<postings::step_kind>(steps[i].kind),
                            steps[i].first_run, steps[i].n_runs,
                            steps[i].complement != 0, steps[i].exclude_run};
  }
  return out;
}

} // namespace

extern "C" cuvsError_t cuvsPostingsWorkspaceSize(const cuvsPostingsStep *steps,
                                                 int64_t n_steps,
                                                 int64_t n_runs, int64_t n_rows,
                                                 int64_t *words) {
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(words != nullptr, "cuvsPostingsWorkspaceSize: null output");
    auto program = to_steps(steps, n_steps);
    *words = postings::workspace_size(
        raft::make_host_vector_view<const postings::step, int64_t>(
            program.data(), n_steps),
        n_runs, n_rows);
  });
}

extern "C" cuvsError_t cuvsPostingsEvaluate(
    cuvsResources_t res, const cuvsPostingsStep *steps, int64_t n_steps,
    const cuvsPostingsRun *runs, int64_t n_runs, DLManagedTensor *const *lists,
    uint32_t n_lists, DLManagedTensor *bitset, int64_t n_rows,
    int64_t first_row, DLManagedTensor *workspace, DLManagedTensor *count) {
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources *>(res);
    RAFT_EXPECTS(n_runs >= 0 && (runs != nullptr || n_runs == 0),
                 "cuvsPostingsEvaluate: null runs");
    RAFT_EXPECTS(lists != nullptr || n_lists == 0,
                 "cuvsPostingsEvaluate: null lists");
    RAFT_EXPECTS(bitset != nullptr, "cuvsPostingsEvaluate: null bitset");
    RAFT_EXPECTS(n_rows >= 0, "cuvsPostingsEvaluate: negative n_rows");
    auto program = to_steps(steps, n_steps);
    std::vector<postings::run> run_list(n_runs);
    for (int64_t r = 0; r < n_runs; r++) {
      run_list[r] = postings::run{runs[r].list, runs[r].first, runs[r].count};
    }
    using list_type = raft::device_vector_view<const uint32_t, int64_t>;
    std::vector<list_type> list_views;
    list_views.reserve(n_lists);
    for (uint32_t i = 0; i < n_lists; i++) {
      RAFT_EXPECTS(
          lists[i] != nullptr &&
              cuvs::core::is_dlpack_device_compatible(lists[i]->dl_tensor),
          "cuvsPostingsEvaluate: list %u is not a device tensor", i);
      list_views.push_back(cuvs::core::from_dlpack<list_type>(lists[i]));
    }
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(bitset->dl_tensor),
                 "cuvsPostingsEvaluate: the bitset must be in device memory");
    auto words =
        cuvs::core::from_dlpack<raft::device_vector_view<uint32_t, int64_t>>(
            bitset);
    RAFT_EXPECTS(words.extent(0) >= (n_rows + 31) / 32,
                 "cuvsPostingsEvaluate: a bitset of %ld words for %ld rows",
                 static_cast<long>(words.extent(0)), static_cast<long>(n_rows));
    std::optional<raft::device_vector_view<uint32_t, int64_t>> workspace_view;
    if (workspace != nullptr) {
      RAFT_EXPECTS(
          cuvs::core::is_dlpack_device_compatible(workspace->dl_tensor),
          "cuvsPostingsEvaluate: the workspace must be in device memory");
      workspace_view =
          cuvs::core::from_dlpack<raft::device_vector_view<uint32_t, int64_t>>(
              workspace);
    }
    std::optional<raft::device_scalar_view<int64_t>> count_view;
    if (count != nullptr) {
      RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(count->dl_tensor),
                   "cuvsPostingsEvaluate: the count must be in device memory");
      auto v =
          cuvs::core::from_dlpack<raft::device_vector_view<int64_t, int64_t>>(
              count);
      RAFT_EXPECTS(v.extent(0) >= 1,
                   "cuvsPostingsEvaluate: an empty count tensor");
      count_view = raft::make_device_scalar_view<int64_t>(v.data_handle());
    }
    postings::evaluate(
        *res_ptr,
        raft::make_host_vector_view<const postings::step, int64_t>(
            program.data(), n_steps),
        raft::make_host_vector_view<const postings::run, int64_t>(
            run_list.data(), n_runs),
        raft::make_host_vector_view<const list_type, int64_t>(
            list_views.data(), static_cast<int64_t>(n_lists)),
        cuvs::core::bitset_view<uint32_t, int64_t>(words.data_handle(), n_rows),
        first_row, workspace_view, count_view);
  });
}
