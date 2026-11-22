/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 * @defgroup core_c_postings C API for bitsets from postings lists
 * @{
 */

/**
 * Defined when cuvsPostingsEvaluate and cuvsPostingsWorkspaceSize are
 * available: a filter over postings lists evaluated on the GPU into a bitset
 * (cuvs::core::postings::evaluate).
 */
#define CUVS_HAS_POSTINGS_EVALUATE 1

/** What a step of a postings program does. */
typedef enum {
  /** Push a leaf: the rows of its runs, then `complement` and `exclude_run`. */
  CUVS_POSTINGS_LEAF = 0,
  /** Pop two results, push the rows in both. */
  CUVS_POSTINGS_AND = 1,
  /** Pop two results, push the rows in either. */
  CUVS_POSTINGS_OR = 2,
  /** Replace the top result by the rows not in it. */
  CUVS_POSTINGS_NOT = 3,
  /** Push every row. */
  CUVS_POSTINGS_ALL = 4,
  /** Push no row. */
  CUVS_POSTINGS_NONE = 5
} cuvsPostingsStepKind;

/** One run of a postings list: the rows `lists[list][first, first + count)`. */
typedef struct {
  uint32_t list;
  int64_t first;
  int64_t count;
} cuvsPostingsRun;

/** One step of a postfix program; the fields after `kind` are for leaves. */
typedef struct {
  cuvsPostingsStepKind kind;
  /** The leaf's runs are `runs[first_run, first_run + n_runs)`. */
  int64_t first_run;
  int64_t n_runs;
  /** Nonzero: the rows in none of its runs instead. */
  int32_t complement;
  /** When 0 or more, the rows of `runs[exclude_run]` are removed last; -1 for
   * none. */
  int64_t exclude_run;
} cuvsPostingsStep;

/**
 * @brief Device scratch cuvsPostingsEvaluate needs, in 32-bit words.
 *
 * @param[in] steps the program
 * @param[in] n_steps its length
 * @param[in] n_runs the number of runs
 * @param[in] n_rows the rows of the bitset written
 * @param[out] words the length of a workspace cuvsPostingsEvaluate takes
 * @return cuvsError_t (an error for a malformed program)
 */
CUVS_EXPORT cuvsError_t cuvsPostingsWorkspaceSize(const cuvsPostingsStep *steps,
                                                  int64_t n_steps,
                                                  int64_t n_runs,
                                                  int64_t n_rows,
                                                  int64_t *words);

/**
 * @brief Run a postfix program over postings lists into a bitset, on the GPU.
 *
 * Bit `i` of `bitset` is set when row `first_row + i` passes, for the rows
 * `[first_row, first_row + n_rows)`; rows of the lists outside them are
 * ignored. Everything runs on the stream of `res` with no wait; the steps and
 * runs are copied before the call returns. See cuvs::core::postings::evaluate.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] steps the program, in postfix order
 * @param[in] n_steps its length
 * @param[in] runs the runs the leaves name
 * @param[in] n_runs their number
 * @param[in] lists `n_lists` 1-D uint32 device tensors: the postings lists (row
 * numbers)
 * @param[in] n_lists their number
 * @param[out] bitset 1-D uint32 device tensor of at least (n_rows + 31) / 32
 * words
 * @param[in] n_rows the rows of the bitset
 * @param[in] first_row the row bit 0 stands for
 * @param[in] workspace 1-D uint32 device tensor of at least
 * cuvsPostingsWorkspaceSize words, 8-byte aligned, or NULL to allocate from the
 * workspace memory of `res`
 * @param[out] count 1-D int64 device tensor of one element receiving the rows
 * that pass, or NULL
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsPostingsEvaluate(
    cuvsResources_t res, const cuvsPostingsStep *steps, int64_t n_steps,
    const cuvsPostingsRun *runs, int64_t n_runs, DLManagedTensor *const *lists,
    uint32_t n_lists, DLManagedTensor *bitset, int64_t n_rows,
    int64_t first_row, DLManagedTensor *workspace, DLManagedTensor *count);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
