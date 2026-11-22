/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/bitset.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <optional>

namespace CUVS_EXPORT cuvs {
namespace core {
namespace postings {

/**
 * @defgroup postings Bitsets from postings lists
 * @{
 *
 * A postings list holds, for each value of an attribute, the rows that hold it, one value's rows
 * after another: the rows of one value, or of a range of values, are one run of the list. A
 * filter over such attributes is a postfix program whose leaves are unions of runs; `evaluate`
 * runs it on the GPU into a bitset of the rows that pass, for use as a
 * `cuvs::neighbors::filtering::bitset_filter`. Its work is proportional to the rows the leaves
 * name plus one pass over the bitset a step, so a filter that names few rows of a large index is
 * built in microseconds, without reading the attribute of every row.
 *
 * The program is planned by the caller, which maps values to runs (from its sorted values and
 * their offsets into the list) and chooses how a leaf is expressed: a leaf naming most of a
 * single-valued attribute's rows is cheaper as the complement of the other values' runs.
 */

/** One run of a postings list: the rows `lists[list][first, first + count)`. */
struct run {
  /** The postings list the run is in (an index into `evaluate`'s `lists`). */
  uint32_t list = 0;
  /** Offset of the run's first row in the list. */
  int64_t first = 0;
  /** Rows in the run. */
  int64_t count = 0;
};

/** What a step of a program does. */
enum class step_kind : int {
  /** Push a leaf: the rows of its runs, then `complement` and `exclude_run`. */
  LEAF = 0,
  /** Pop two results, push the rows in both. */
  AND = 1,
  /** Pop two results, push the rows in either. */
  OR = 2,
  /** Replace the top result by the rows not in it. */
  NOT = 3,
  /** Push every row. */
  ALL = 4,
  /** Push no row. */
  NONE = 5
};

/** One step of a postfix program. */
struct step {
  step_kind kind = step_kind::LEAF;
  /** LEAF: its runs are `runs[first_run, first_run + n_runs)`. */
  int64_t first_run = 0;
  /** LEAF: the number of its runs; 0 for a leaf of no rows. */
  int64_t n_runs = 0;
  /** LEAF: the rows in none of its runs instead. */
  bool complement = false;
  /**
   * LEAF: when 0 or more, the rows of `runs[exclude_run]` are removed last, after `complement`.
   * For example, a negated comparison that must not pass rows whose value is NULL is the
   * complement of the compared values' runs with the NULL value's run excluded.
   */
  int64_t exclude_run = -1;
};

/**
 * @brief Device scratch `evaluate` needs, in 32-bit words.
 *
 * One bitset of `n_rows` rows for each result the program holds at once beyond its first, and
 * the uploaded runs.
 *
 * @param[in] program the steps, as for `evaluate`
 * @param[in] n_runs the number of runs, as for `evaluate`
 * @param[in] n_rows the rows of the bitset `evaluate` writes
 * @return the length of a `workspace` that `evaluate` takes
 */
auto workspace_size(raft::host_vector_view<const step, int64_t> program,
                    int64_t n_runs,
                    int64_t n_rows) -> int64_t;

/**
 * @brief Run a postfix program over postings lists into a bitset.
 *
 * Bit `i` of `out` is set when row `first_row + i` passes the program, for the rows
 * `[first_row, first_row + out.size())`: the whole index, or the rows of one part of it, such as
 * a shard. Rows of the lists outside that window are ignored; within a LEAF, rows that are in
 * several of its runs are set once. The program must leave one result; an AND or OR needs two
 * before it and a NOT one.
 *
 * Everything runs on the stream of @p res, with no wait: @p out, @p count and @p workspace are
 * ready for same-stream use. The runs are copied to the device before the call returns, so the
 * host arrays can be released at once; the lists must stay unchanged until the stream reaches
 * the work.
 *
 * Usage example:
 * @code{.cpp}
 * namespace postings = cuvs::core::postings;
 * // Rows whose value is 3 or 7 and not 0: values 3 and 7 are runs 0 and 1 of list 0 (the rows
 * // of attribute a), value 0 run 2 of list 1 (attribute b).
 * std::vector<postings::run> runs = {{0, offset_3, count_3}, {0, offset_7, count_7},
 *                                    {1, offset_0, count_0}};
 * std::vector<postings::step> program(3);
 * program[0] = {postings::step_kind::LEAF, 0, 2};
 * program[1] = {postings::step_kind::LEAF, 2, 1, true};  // b <> 0
 * program[2].kind = postings::step_kind::AND;
 * postings::evaluate(res,
 *                    raft::make_host_vector_view<const postings::step, int64_t>(program.data(), 3),
 *                    raft::make_host_vector_view<const postings::run, int64_t>(runs.data(), 3),
 *                    raft::make_host_vector_view<const raft::device_vector_view<const uint32_t,
 *                    int64_t>, int64_t>(lists.data(), 2),
 *                    bitset.view());
 * @endcode
 *
 * @param[in] res RAFT resources: the stream, and the workspace memory when @p workspace is not
 * given
 * @param[in] program the steps, in postfix order
 * @param[in] runs the runs the leaves name
 * @param[in] lists the postings lists, in device memory: row numbers of the index, below 2^32
 * @param[out] out the bitset of the rows `[first_row, first_row + out.size())`; every word is
 * written
 * @param[in] first_row the row bit 0 of @p out stands for
 * @param[in] workspace device scratch of at least `workspace_size(program, runs.size(),
 * out.size())` words; allocated from the workspace memory of @p res when not given
 * @param[out] count when given, receives the number of rows that pass
 */
void evaluate(
  raft::resources const& res,
  raft::host_vector_view<const step, int64_t> program,
  raft::host_vector_view<const run, int64_t> runs,
  raft::host_vector_view<const raft::device_vector_view<const uint32_t, int64_t>, int64_t> lists,
  cuvs::core::bitset_view<uint32_t, int64_t> out,
  int64_t first_row                                                    = 0,
  std::optional<raft::device_vector_view<uint32_t, int64_t>> workspace = std::nullopt,
  std::optional<raft::device_scalar_view<int64_t>> count               = std::nullopt);

/** @} */

}  // namespace postings
}  // namespace core
}  // namespace CUVS_EXPORT cuvs
