/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/core/interop.hpp"

#include <cuvs/core/c_api.h>
#include <cuvs/core/postings.h>
#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <raft/core/device_mdspan.hpp>

#include <cstdint>
#include <vector>

namespace {

template <typename T>
auto device_vector_tensor(T *data, int64_t size) -> DLManagedTensor {
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_vector_view<T, int64_t>(data, size),
                        &tensor);
  return tensor;
}

template <typename T> auto device_copy(std::vector<T> const &host) -> T * {
  T *data = nullptr;
  EXPECT_EQ(cudaMalloc(&data, sizeof(T) * (host.empty() ? 1 : host.size())),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(data, host.data(), sizeof(T) * host.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  return data;
}

} // namespace

// Rows 0-99. Attribute a: value 0 is rows 0-49, value 1 rows 50-99 (list 0 in
// value order). Attribute b, multi-valued: value 0 is the even rows, value 1
// the multiples of 3 (list 1). "a = 1 AND NOT (b = 0)" passes the odd rows
// 51-99; over the window from row 60, bit i is row 60+i.
TEST(PostingsC, Evaluate) {
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  std::vector<uint32_t> a, b;
  for (uint32_t r = 0; r < 100; r++) {
    a.push_back(r);
  }
  for (uint32_t r = 0; r < 100; r += 2) {
    b.push_back(r);
  }
  int64_t evens = static_cast<int64_t>(b.size());
  for (uint32_t r = 0; r < 100; r += 3) {
    b.push_back(r);
  }
  uint32_t *a_dev = device_copy(a);
  uint32_t *b_dev = device_copy(b);
  DLManagedTensor lists_tensor[2] = {
      device_vector_tensor(a_dev, 100),
      device_vector_tensor(b_dev, static_cast<int64_t>(b.size()))};
  DLManagedTensor *lists[2] = {&lists_tensor[0], &lists_tensor[1]};

  cuvsPostingsRun runs[2] = {{0, 50, 50}, {1, 0, evens}};
  cuvsPostingsStep steps[4] = {{CUVS_POSTINGS_LEAF, 0, 1, 0, -1},
                               {CUVS_POSTINGS_LEAF, 1, 1, 0, -1},
                               {CUVS_POSTINGS_NOT, 0, 0, 0, -1},
                               {CUVS_POSTINGS_AND, 0, 0, 0, -1}};

  const int64_t first_row = 60, n_rows = 40;
  int64_t words = 0;
  ASSERT_EQ(cuvsPostingsWorkspaceSize(steps, 4, 2, n_rows, &words),
            CUVS_SUCCESS);
  EXPECT_GT(words, 0);

  uint32_t *bits_dev = device_copy(std::vector<uint32_t>(2, 0xffffffffu));
  int64_t *count_dev = device_copy(std::vector<int64_t>(1, -1));
  uint32_t *ws_dev = device_copy(std::vector<uint32_t>(words, 0));
  DLManagedTensor bits = device_vector_tensor(bits_dev, 2);
  DLManagedTensor count = device_vector_tensor(count_dev, 1);
  DLManagedTensor ws = device_vector_tensor(ws_dev, words);

  for (DLManagedTensor *workspace :
       {&ws, static_cast<DLManagedTensor *>(nullptr)}) {
    ASSERT_EQ(cuvsPostingsEvaluate(res, steps, 4, runs, 2, lists, 2, &bits,
                                   n_rows, first_row, workspace, &count),
              CUVS_SUCCESS)
        << cuvsGetLastErrorText();
    ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);
    std::vector<uint32_t> host(2);
    int64_t found = 0;
    ASSERT_EQ(cudaMemcpy(host.data(), bits_dev, 8, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&found, count_dev, 8, cudaMemcpyDeviceToHost),
              cudaSuccess);
    int64_t expected = 0;
    for (int64_t i = 0; i < 64; i++) {
      bool want = i < n_rows && (first_row + i) % 2 == 1;
      expected += want;
      EXPECT_EQ(((host[i / 32] >> (i % 32)) & 1u) != 0, want) << "bit " << i;
    }
    EXPECT_EQ(found, expected);
  }

  // A malformed program is an error, with a message.
  cuvsPostingsStep bad[1] = {{CUVS_POSTINGS_AND, 0, 0, 0, -1}};
  EXPECT_EQ(cuvsPostingsWorkspaceSize(bad, 1, 0, n_rows, &words), CUVS_ERROR);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(bits_dev);
  cudaFree(count_dev);
  cudaFree(ws_dev);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}
