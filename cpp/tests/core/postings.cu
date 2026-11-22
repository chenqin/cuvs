/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/postings.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace cuvs::core::postings {

namespace {

using list_view = raft::device_vector_view<const uint32_t, int64_t>;

/** A postings list and the runs of its values: value v's rows are rows[offsets[v], offsets[v+1]).
 */
struct attribute {
  std::vector<uint32_t> rows;
  std::vector<int64_t> offsets;
};

/** Rows 0..n_rows-1, each holding `per_row` values (repeats possible) of n_values, as postings. */
auto make_attribute(std::mt19937& rng, int64_t n_rows, int n_values, int per_row) -> attribute
{
  std::vector<std::vector<uint32_t>> by_value(n_values);
  std::uniform_int_distribution<int> pick(0, n_values - 1);
  for (int64_t r = 0; r < n_rows; r++) {
    std::vector<int> held;
    for (int j = 0; j < per_row; j++) {
      held.push_back(pick(rng));
    }
    std::sort(held.begin(), held.end());
    held.erase(std::unique(held.begin(), held.end()), held.end());
    for (int v : held) {
      by_value[v].push_back(static_cast<uint32_t>(r));
    }
  }
  attribute a;
  a.offsets.push_back(0);
  for (auto& rows : by_value) {
    a.rows.insert(a.rows.end(), rows.begin(), rows.end());
    a.offsets.push_back(static_cast<int64_t>(a.rows.size()));
  }
  return a;
}

/** A random program and its runs over the attributes, and the rows it passes, on the host. */
struct case_t {
  std::vector<step> program;
  std::vector<run> runs;
  std::vector<char> expected;  // over all rows of the index
};

auto make_case(std::mt19937& rng, std::vector<attribute> const& attrs, int64_t n_rows, int leaves)
  -> case_t
{
  case_t c;
  std::vector<std::vector<char>> stack;
  std::uniform_int_distribution<int> coin(0, 1);
  std::uniform_int_distribution<int> op(0, 9);
  auto rows_of = [&](run const& x) {
    std::vector<char> in(n_rows, 0);
    for (int64_t i = x.first; i < x.first + x.count; i++) {
      in[attrs[x.list].rows[i]] = 1;
    }
    return in;
  };
  int pushed = 0;
  while (pushed < leaves || stack.size() > 1) {
    int o = op(rng);
    if (pushed < leaves && (stack.size() < 2 || o < 5)) {
      step s;
      int kind = std::uniform_int_distribution<int>(0, 11)(rng);
      if (kind == 0) {
        s.kind = step_kind::ALL;
        stack.emplace_back(n_rows, 1);
      } else if (kind == 1) {
        s.kind = step_kind::NONE;
        stack.emplace_back(n_rows, 0);
      } else {
        uint32_t list = std::uniform_int_distribution<uint32_t>(0, attrs.size() - 1)(rng);
        auto const& a = attrs[list];
        int n_values  = static_cast<int>(a.offsets.size()) - 1;
        std::vector<char> in(n_rows, 0);
        s.first_run = static_cast<int64_t>(c.runs.size());
        // Single values and ranges of values, some of them empty.
        int n = std::uniform_int_distribution<int>(0, 4)(rng);
        for (int k = 0; k < n; k++) {
          int v0 = std::uniform_int_distribution<int>(0, n_values - 1)(rng);
          int v1 = std::min(n_values, v0 + std::uniform_int_distribution<int>(0, 3)(rng));
          run x{list, a.offsets[v0], a.offsets[v1] - a.offsets[v0]};
          c.runs.push_back(x);
          auto r = rows_of(x);
          for (int64_t i = 0; i < n_rows; i++) {
            in[i] |= r[i];
          }
        }
        s.n_runs     = n;
        s.complement = coin(rng);
        if (s.complement) {
          for (auto& b : in) {
            b = !b;
          }
        }
        if (coin(rng)) {
          // Exclude one value's rows: sometimes a run of the leaf's own.
          if (n > 0 && coin(rng)) {
            s.exclude_run = s.first_run + std::uniform_int_distribution<int>(0, n - 1)(rng);
          } else {
            int v         = std::uniform_int_distribution<int>(0, n_values - 1)(rng);
            s.exclude_run = static_cast<int64_t>(c.runs.size());
            c.runs.push_back(run{list, a.offsets[v], a.offsets[v + 1] - a.offsets[v]});
          }
          auto r = rows_of(c.runs[s.exclude_run]);
          for (int64_t i = 0; i < n_rows; i++) {
            in[i] &= !r[i];
          }
        }
        stack.push_back(std::move(in));
      }
      c.program.push_back(s);
      pushed++;
    } else if (o == 9) {
      step s;
      s.kind = step_kind::NOT;
      for (auto& b : stack.back()) {
        b = !b;
      }
      c.program.push_back(s);
    } else {
      step s;
      s.kind = coin(rng) ? step_kind::AND : step_kind::OR;
      auto b = std::move(stack.back());
      stack.pop_back();
      auto& a = stack.back();
      for (int64_t i = 0; i < n_rows; i++) {
        a[i] = s.kind == step_kind::AND ? (a[i] && b[i]) : (a[i] || b[i]);
      }
      c.program.push_back(s);
    }
  }
  c.expected = std::move(stack.back());
  return c;
}

class PostingsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    std::mt19937 rng(42);
    // A single-valued attribute, a multi-valued one with repeats, and one of few values.
    attrs.push_back(make_attribute(rng, n_rows, 97, 1));
    attrs.push_back(make_attribute(rng, n_rows, 211, 4));
    attrs.push_back(make_attribute(rng, n_rows, 3, 1));
    auto stream = raft::resource::get_cuda_stream(res);
    for (auto const& a : attrs) {
      device_lists.push_back(
        raft::make_device_vector<uint32_t, int64_t>(res, static_cast<int64_t>(a.rows.size())));
      raft::copy(device_lists.back().data_handle(), a.rows.data(), a.rows.size(), stream);
    }
    for (auto const& d : device_lists) {
      lists.push_back(raft::make_const_mdspan(d.view()));
    }
  }

  /** Evaluate c over rows [first_row, first_row + n) and compare with the host's answer. */
  void check(case_t const& c, int64_t first_row, int64_t n, bool give_workspace)
  {
    auto stream  = raft::resource::get_cuda_stream(res);
    auto program = raft::make_host_vector_view<const step, int64_t>(
      c.program.data(), static_cast<int64_t>(c.program.size()));
    auto runs = raft::make_host_vector_view<const run, int64_t>(
      c.runs.data(), static_cast<int64_t>(c.runs.size()));
    auto lists_view = raft::make_host_vector_view<const list_view, int64_t>(
      lists.data(), static_cast<int64_t>(lists.size()));
    int64_t words = (n + 31) / 32;
    // Garbage in the output and the workspace: every word must be written.
    auto bits = raft::make_device_vector<uint32_t, int64_t>(res, std::max<int64_t>(words, 1));
    RAFT_CUDA_TRY(cudaMemsetAsync(bits.data_handle(), 0xa5, bits.size() * 4, stream.get()));
    auto count = raft::make_device_scalar<int64_t>(res, -1);
    std::optional<raft::device_vector<uint32_t, int64_t>> ws;
    std::optional<raft::device_vector_view<uint32_t, int64_t>> ws_view;
    if (give_workspace) {
      ws.emplace(raft::make_device_vector<uint32_t, int64_t>(
        res, workspace_size(program, runs.extent(0), n)));
      RAFT_CUDA_TRY(cudaMemsetAsync(ws->data_handle(), 0x5a, ws->size() * 4, stream.get()));
      ws_view = ws->view();
    }
    evaluate(res,
             program,
             runs,
             lists_view,
             cuvs::core::bitset_view<uint32_t, int64_t>(bits.data_handle(), n),
             first_row,
             ws_view,
             count.view());
    std::vector<uint32_t> host(words);
    int64_t found = 0;
    raft::copy(host.data(), bits.data_handle(), words, stream);
    raft::copy(&found, count.data_handle(), 1, stream);
    raft::resource::sync_stream(res);
    int64_t expected_count = 0;
    for (int64_t i = 0; i < n; i++) {
      bool want = c.expected[first_row + i];
      bool got  = (host[i / 32] >> (i % 32)) & 1u;
      expected_count += want;
      ASSERT_EQ(got, want) << "row " << first_row + i;
    }
    for (int64_t i = n; i < words * 32; i++) {
      ASSERT_FALSE((host[i / 32] >> (i % 32)) & 1u) << "bit " << i << " past the rows";
    }
    EXPECT_EQ(found, expected_count);
  }

  static constexpr int64_t n_rows = 10007;
  raft::device_resources res;
  std::vector<attribute> attrs;
  std::vector<raft::device_vector<uint32_t, int64_t>> device_lists;
  std::vector<list_view> lists;
};

// Random programs of every step kind match a host evaluation, over the whole index and over
// windows of it (as for the shards of an index), with and without a caller's workspace.
TEST_F(PostingsTest, MatchesHostEvaluation)
{
  std::mt19937 rng(7);
  for (int t = 0; t < 200; t++) {
    auto c = make_case(rng, attrs, n_rows, 1 + t % 6);
    check(c, 0, n_rows, t % 2 == 0);
    check(c, 3001, 4000, t % 2 == 1);
    check(c, n_rows - 33, 33, false);
  }
}

// A leaf of no runs is no row, and its complement every row of the window.
TEST_F(PostingsTest, EmptyLeaf)
{
  case_t c;
  c.program.push_back(step{});
  c.expected.assign(n_rows, 0);
  check(c, 0, n_rows, false);
  c.program[0].complement = true;
  c.expected.assign(n_rows, 1);
  check(c, 0, n_rows, false);
  check(c, 5, 64, true);
}

// Malformed programs and runs are refused, saying why.
TEST_F(PostingsTest, Refusals)
{
  auto expect_refused = [](auto&& f, std::string const& why) {
    try {
      f();
      FAIL() << "not refused: " << why;
    } catch (raft::logic_error const& e) {
      EXPECT_NE(std::string(e.what()).find(why), std::string::npos) << e.what();
    }
  };
  auto bits       = raft::make_device_vector<uint32_t, int64_t>(res, (n_rows + 31) / 32);
  auto out        = cuvs::core::bitset_view<uint32_t, int64_t>(bits.data_handle(), n_rows);
  auto lists_view = raft::make_host_vector_view<const list_view, int64_t>(
    lists.data(), static_cast<int64_t>(lists.size()));
  auto run_with = [&](std::vector<step> const& p, std::vector<run> const& r) {
    evaluate(res,
             raft::make_host_vector_view<const step, int64_t>(p.data(), p.size()),
             raft::make_host_vector_view<const run, int64_t>(r.data(), r.size()),
             lists_view,
             out);
  };
  step leaf{step_kind::LEAF, 0, 1};
  step both{step_kind::AND};
  expect_refused([&] { run_with({leaf, both}, {run{0, 0, 1}}); }, "combines two results of 1");
  expect_refused([&] { run_with({leaf, leaf}, {run{0, 0, 1}}); }, "leaves 2 results");
  expect_refused([&] { run_with({leaf}, {run{9, 0, 1}}); }, "is in list 9");
  expect_refused([&] { run_with({leaf}, {run{0, 0, n_rows + 1}}); }, "covers [0,");
  expect_refused([&] { run_with({step{step_kind::LEAF, 0, 2}}, {run{0, 0, 1}}); }, "names runs");
  std::vector<step> p = {leaf};
  std::vector<run> r  = {run{0, 0, 1}};
  auto small          = raft::make_device_vector<uint32_t, int64_t>(res, 1);
  expect_refused(
    [&] {
      evaluate(res,
               raft::make_host_vector_view<const step, int64_t>(p.data(), 1),
               raft::make_host_vector_view<const run, int64_t>(r.data(), 1),
               lists_view,
               out,
               0,
               small.view());
    },
    "a workspace of 1 words");
}

}  // namespace

}  // namespace cuvs::core::postings
