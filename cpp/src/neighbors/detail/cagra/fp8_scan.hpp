/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/preprocessing/quantize/fp8.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>

/*
 * The FP8 filtered scan of CAGRA search (search_params::filtered_route): every listed row is scored
 * once from its FP8 codes for all queries of the call, each query keeps its best candidates, and
 * those are re-scored from the index's FP16 rows. Defined in fp8_scan.cu.
 */
namespace cuvs::neighbors::cagra::detail::fp8_scan {

/** Dimensions the scan takes, at most. */
inline constexpr uint32_t max_dims = 512;
/** Neighbors (k) the scan returns per query, at most. */
inline constexpr int64_t max_k = 4096;

/** Whether this build carries scan kernels for the device of res (an sm_89 or newer image). */
[[nodiscard]] CUVS_EXPORT auto is_supported(raft::resources const& res) -> bool;

/** The FP16 rows the candidates are re-scored from: n_rows rows of dim values, stride apart. */
struct rerank_rows {
  const half* data = nullptr;
  int64_t n_rows   = 0;
  uint32_t dim     = 0;
  int64_t stride   = 0;
};

/**
 * The k nearest listed rows to each query. Queries are processed in groups of up to 64; for each
 * group, with c candidates per query (params.scan_candidates, or k + max(16, 3k), within the
 * listed rows and 4096):
 *
 * 1. Fused selection (65536 rows or more, when the cutoff rank and its buffers fit): score a spread
 *    sample of 8192 of the rows, take each query's cutoff at the rank that should leave about 4c
 *    rows under it (rank 32 at least), scan the rows once more keeping the distances at or under
 *    the cutoff (32768 a query at most), and select the c best of those. When params.scan_retry is
 *    set, the call does not wait: it writes 1 to the flag of each query that kept fewer than c or
 *    more than 32768 rows (its candidates are then not the c best; search it again with
 *    params.scan_full_selection) and 0 to every other query. Otherwise it waits for the kept
 *    counts and runs step 2 for the whole group when any is out of range.
 * 2. Full selection (fewer rows, params.scan_full_selection, or a failed cutoff): score every row
 *    and select the c best.
 * 3. Re-score the c candidates from the FP16 rows and keep each query's k best.
 *
 * Candidates tied in FP8 distance at the cut of step 1 are kept in no fixed order.
 *
 * @tparam OutputIdxT uint32_t or int64_t (instantiated in fp8_scan.cu)
 * @param[in] res raft resources
 * @param[in] params reads scan_candidates, scan_skip_last_block, scan_retry and
 *   scan_full_selection
 * @param[in] codes FP8 codes of the dataset rows, encoded_row_length() >= dim rounded up to 16
 * @param[in] dataset FP16 rows of the index
 * @param[in] queries [n_queries, dim]
 * @param[in] rows n_rows unique row numbers on the device, each below dataset.n_rows
 * @param[in] n_rows rows listed
 * @param[out] neighbors [n_queries, k] row numbers of the index; slots past the rows found hold
 *   0xFFFFFFFF (as OutputIdxT: 4294967295 in an int64 output)
 * @param[out] distances [n_queries, k] squared L2 distances; FLT_MAX past the rows found
 */
template <typename OutputIdxT>
CUVS_EXPORT void search(raft::resources const& res,
                        const cagra::search_params& params,
                        const cuvs::preprocessing::quantize::fp8::quantizer_view& codes,
                        const rerank_rows& dataset,
                        raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
                        const uint32_t* rows,
                        uint32_t n_rows,
                        raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
                        raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** The rows a bitset of n_rows bits passes, in increasing order. Waits on the stream. */
[[nodiscard]] CUVS_EXPORT auto bitset_rows(raft::resources const& res,
                                           const uint32_t* words,
                                           int64_t n_rows) -> rmm::device_uvector<uint32_t>;

/** Every row number below n_rows, in order. */
[[nodiscard]] CUVS_EXPORT auto all_rows(raft::resources const& res, int64_t n_rows)
  -> rmm::device_uvector<uint32_t>;

}  // namespace cuvs::neighbors::cagra::detail::fp8_scan
