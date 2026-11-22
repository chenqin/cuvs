/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Builds, saves, loads and searches the two CAGRA indexes that serve heavily filtered searches:
//
//  - a graph index of FP16 rows, searched with the two-hop walk (filter_expansion PASSING_TWO_HOP),
//    which suits filters that pass a few percent of the rows or more;
//  - a graph-free index of the same FP16 rows with FP8 codes attached, searched by the filtered
//  scan
//    (filtered_route SCAN), which suits narrow filters and is near exact.
//
// A graph index can hold the codes as well and let filtered_route AUTO choose per search. The codes
// are not serialized with the index, so this example writes them to their own file and attaches
// them again after loading. See "Building indexes for filtered search" in the CAGRA guide.
//
// Usage:
//   CAGRA_FILTERED_INDEXES_EXAMPLE                         synthetic rows, build, save, load,
//   search CAGRA_FILTERED_INDEXES_EXAMPLE build DATA.fbin OUT     build both indexes from float32
//   rows CAGRA_FILTERED_INDEXES_EXAMPLE search OUT              load them and search at 1% and 10%
//   filters
//
// DATA.fbin is the benchmark format: uint32 rows, uint32 dimension, then float32 rows. OUT receives
// graph.cagra (FP16 rows and graph), rows.f16 (the FP16 rows alone, for the graph-free index) and
// codes.fp8 (FP8 codes and row scales).

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/preprocessing/quantize/fp8.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

namespace cagra = cuvs::neighbors::cagra;
namespace fp8   = cuvs::preprocessing::quantize::fp8;

constexpr int64_t n_queries = 100;
constexpr int64_t top_k     = 100;

/** Host FP16 rows, read from float32 (.fbin) or made up. */
struct host_rows {
  int64_t n_rows = 0;
  int64_t dim    = 0;
  std::vector<half> data;
};

auto read_fbin_as_half(std::string const& path) -> host_rows
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot open " + path); }
  uint32_t header[2];
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  host_rows rows{header[0], header[1], {}};
  rows.data.resize(static_cast<size_t>(rows.n_rows) * rows.dim);
  std::vector<float> chunk(static_cast<size_t>(rows.dim) * 65536);
  for (int64_t first = 0; first < rows.n_rows; first += 65536) {
    int64_t n = std::min<int64_t>(65536, rows.n_rows - first);
    in.read(reinterpret_cast<char*>(chunk.data()), n * rows.dim * sizeof(float));
    for (int64_t i = 0; i < n * rows.dim; i++) {
      rows.data[first * rows.dim + i] = __float2half(chunk[i]);
    }
  }
  if (!in) { throw std::runtime_error(path + " ends early"); }
  return rows;
}

auto synthetic_rows(raft::device_resources const& res, int64_t n_rows, int64_t dim) -> host_rows
{
  auto values = raft::make_device_vector<float, int64_t>(res, n_rows * dim);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(res, rng, values.data_handle(), values.size(), -1.0f, 1.0f);
  std::vector<float> host(values.size());
  raft::copy(host.data(), values.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  host_rows rows{n_rows, dim, std::vector<half>(host.size())};
  std::transform(
    host.begin(), host.end(), rows.data.begin(), [](float v) { return __float2half(v); });
  return rows;
}

auto to_device(raft::device_resources const& res, host_rows const& rows)
  -> raft::device_matrix<half, int64_t>
{
  auto out = raft::make_device_matrix<half, int64_t>(res, rows.n_rows, rows.dim);
  raft::copy(
    out.data_handle(), rows.data.data(), rows.data.size(), raft::resource::get_cuda_stream(res));
  return out;
}

/**
 * CAGRA's rows: a view of the matrix when its rows are already as wide as CAGRA needs (64 FP16
 * values are), else an owning padded copy, which `owner` keeps alive.
 */
struct padded_rows {
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>> owner;
  cuvs::neighbors::device_padded_dataset_view<half, int64_t> view;
};

auto make_padded_rows(raft::device_resources const& res,
                      raft::device_matrix_view<const half, int64_t> rows) -> padded_rows
{
  if (cuvs::neighbors::matrix_row_width_matches_cagra_required(rows)) {
    return {nullptr, cuvs::neighbors::make_device_padded_dataset_view(res, rows)};
  }
  auto owner = cuvs::neighbors::make_device_padded_dataset(res, rows);
  auto view  = owner->view();
  return {std::move(owner), view};
}

template <typename T>
void write_array(std::ofstream& out, T const* data, size_t n)
{
  out.write(reinterpret_cast<char const*>(data), n * sizeof(T));
}

template <typename T>
void read_array(std::ifstream& in, T* data, size_t n)
{
  in.read(reinterpret_cast<char*>(data), n * sizeof(T));
}

/** rows.f16: int64 rows, int64 dimension, then the FP16 rows, densely. */
void save_rows(std::string const& path, host_rows const& rows)
{
  std::ofstream out(path, std::ios::binary);
  int64_t header[2] = {rows.n_rows, rows.dim};
  write_array(out, header, 2);
  write_array(out, rows.data.data(), rows.data.size());
}

auto load_rows(std::string const& path) -> host_rows
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot open " + path); }
  int64_t header[2];
  read_array(in, header, 2);
  host_rows rows{
    header[0], header[1], std::vector<half>(static_cast<size_t>(header[0]) * header[1])};
  read_array(in, rows.data.data(), rows.data.size());
  return rows;
}

/** codes.fp8: int64 rows, int64 code row width, then the codes, then one float scale a row. */
void save_codes(raft::device_resources const& res, std::string const& path, fp8::quantizer const& q)
{
  auto view  = q.view();
  auto rows  = view.codes.extent(0);
  auto width = view.codes.extent(1);
  std::vector<uint8_t> codes(static_cast<size_t>(rows) * width);
  std::vector<float> scales(rows);
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(codes.data(), view.codes.data_handle(), codes.size(), stream);
  raft::copy(scales.data(), view.row_scales.data_handle(), scales.size(), stream);
  raft::resource::sync_stream(res);
  std::ofstream out(path, std::ios::binary);
  int64_t header[2] = {rows, width};
  write_array(out, header, 2);
  write_array(out, codes.data(), codes.size());
  write_array(out, scales.data(), scales.size());
}

/** Device copies of codes.fp8, for index::update_scan_codes. */
struct device_codes {
  raft::device_matrix<uint8_t, int64_t> codes;
  raft::device_vector<float, int64_t> row_scales;

  auto view() const -> fp8::quantizer_view
  {
    return fp8::quantizer_view{raft::make_const_mdspan(codes.view()),
                               raft::make_const_mdspan(row_scales.view())};
  }
};

auto load_codes(raft::device_resources const& res, std::string const& path) -> device_codes
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot open " + path); }
  int64_t header[2];
  read_array(in, header, 2);
  std::vector<uint8_t> codes(static_cast<size_t>(header[0]) * header[1]);
  std::vector<float> scales(header[0]);
  read_array(in, codes.data(), codes.size());
  read_array(in, scales.data(), scales.size());
  device_codes out{raft::make_device_matrix<uint8_t, int64_t>(res, header[0], header[1]),
                   raft::make_device_vector<float, int64_t>(res, header[0])};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(out.codes.data_handle(), codes.data(), codes.size(), stream);
  raft::copy(out.row_scales.data_handle(), scales.data(), scales.size(), stream);
  raft::resource::sync_stream(res);
  return out;
}

/** Build the graph index and the FP8 codes, and write the three files. */
void build(raft::device_resources const& res, host_rows const& rows, std::string const& out_dir)
{
  auto started = std::chrono::steady_clock::now();
  auto dataset = to_device(res, rows);

  // The graph: the settings of the filtered-search benchmark. Degree 32 leaves the two-hop walk
  // about a thousand rows within two hops of each row.
  cagra::index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = 32;
  params.intermediate_graph_degree = 64;
  auto padded                      = make_padded_rows(res, raft::make_const_mdspan(dataset.view()));
  auto index                       = cagra::build(res, params, padded.view);
  cagra::serialize(res, out_dir + "/graph.cagra", index, /*include_dataset=*/true);

  // The codes: one E4M3 code per component and one scale per row, from the same FP16 rows.
  fp8::quantizer codes(res, rows.n_rows, static_cast<uint32_t>(rows.dim));
  fp8::transform(res, raft::make_const_mdspan(dataset.view()), codes);
  save_codes(res, out_dir + "/codes.fp8", codes);
  save_rows(out_dir + "/rows.f16", rows);

  std::chrono::duration<double> took = std::chrono::steady_clock::now() - started;
  std::cout << "built " << rows.n_rows << " rows of " << rows.dim << " dimensions into " << out_dir
            << " in " << took.count() << " s" << std::endl;
}

/** A filter passing about pass_rate of the rows, scattered: the hard case for a graph walk. */
auto make_filter(raft::device_resources const& res, int64_t n_rows, double pass_rate)
  -> std::pair<raft::device_vector<uint32_t, int64_t>, int64_t>
{
  std::vector<uint32_t> words((n_rows + 31) / 32, 0);
  int64_t passing = 0;
  auto threshold  = static_cast<uint32_t>(pass_rate * 4294967295.0);
  for (int64_t i = 0; i < n_rows; i++) {
    if (static_cast<uint32_t>(static_cast<uint64_t>(i) * 2654435761u) <= threshold) {
      words[i / 32] |= 1u << (i % 32);
      passing++;
    }
  }
  auto out = raft::make_device_vector<uint32_t, int64_t>(res, words.size());
  raft::copy(out.data_handle(), words.data(), words.size(), raft::resource::get_cuda_stream(res));
  return {std::move(out), passing};
}

template <typename IdxT>
auto to_host(raft::device_resources const& res, raft::device_matrix<IdxT, int64_t> const& m)
  -> std::vector<IdxT>
{
  std::vector<IdxT> out(m.size());
  raft::copy(out.data(), m.data_handle(), out.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return out;
}

double recall(std::vector<uint32_t> const& found, std::vector<int64_t> const& truth)
{
  int64_t hits = 0, total = 0;
  for (int64_t q = 0; q < n_queries; q++) {
    std::unordered_set<int64_t> expected;
    for (int64_t j = 0; j < top_k; j++) {
      if (truth[q * top_k + j] >= 0) { expected.insert(truth[q * top_k + j]); }
    }
    for (int64_t j = 0; j < top_k; j++) {
      hits += expected.count(found[q * top_k + j]);
    }
    total += static_cast<int64_t>(expected.size());
  }
  return total ? static_cast<double>(hits) / total : 1.0;
}

/** Load both indexes from out_dir and search them with a 1% and a 10% filter. */
void search(raft::device_resources const& res, std::string const& out_dir)
{
  // The graph index, its FP16 rows from the same file, and its codes (for AUTO).
  cagra::index<half, uint32_t> graph(res);
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>> graph_rows;
  cagra::deserialize(res, out_dir + "/graph.cagra", &graph, &graph_rows);
  auto codes = load_codes(res, out_dir + "/codes.fp8");
  graph.update_scan_codes(res, codes.view());

  // The graph-free index: the FP16 rows and the codes, no graph. It can be searched by the scan
  // only, so it holds half the device memory of a graph index whose graph is as large as its rows.
  auto rows      = load_rows(out_dir + "/rows.f16");
  auto rows_dev  = to_device(res, rows);
  auto scan_rows = make_padded_rows(res, raft::make_const_mdspan(rows_dev.view()));
  cagra::index<half, uint32_t> scan_only(res, cuvs::distance::DistanceType::L2Expanded);
  scan_only = cagra::update_dataset(res, std::move(scan_only), scan_rows.view);
  scan_only.update_scan_codes(res, codes.view());

  // Queries: the first rows, so that every query has neighbours at every filter width.
  auto queries = raft::make_device_matrix<half, int64_t>(res, n_queries, rows.dim);
  raft::copy(
    queries.data_handle(), rows.data.data(), queries.size(), raft::resource::get_cuda_stream(res));

  // Exact answers: brute force over the same FP16 rows.
  auto exact_index = cuvs::neighbors::brute_force::build(
    res, cuvs::neighbors::brute_force::index_params{}, raft::make_const_mdspan(rows_dev.view()));

  // Two-hop: the benchmark's settings.
  cagra::search_params two_hop;
  two_hop.algo                 = cagra::search_algo::MULTI_CTA;
  two_hop.filter_expansion     = cagra::filter_expansion_mode::PASSING_TWO_HOP;
  two_hop.filtered_route       = cagra::filtered_route_mode::GRAPH;
  two_hop.itopk_size           = 512;
  two_hop.search_width         = 4;
  two_hop.max_iterations       = 18;
  two_hop.num_random_samplings = 2;
  two_hop.thread_block_size    = 64;

  // The scan: re-score twice k candidates from the FP16 rows.
  cagra::search_params scan;
  scan.filtered_route  = cagra::filtered_route_mode::SCAN;
  scan.scan_candidates = 2 * top_k;

  for (double pass_rate : {0.01, 0.10}) {
    auto [words, passing] = make_filter(res, rows.n_rows, pass_rate);
    cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(
      cuvs::core::bitset_view<uint32_t, int64_t>(words.data_handle(), rows.n_rows));

    auto truth_rows = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
    auto truth_dist = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k);
    cuvs::neighbors::brute_force::search(res,
                                         cuvs::neighbors::brute_force::search_params{},
                                         exact_index,
                                         raft::make_const_mdspan(queries.view()),
                                         truth_rows.view(),
                                         truth_dist.view(),
                                         filter);
    auto truth = to_host(res, truth_rows);

    auto run = [&](char const* name,
                   cagra::index<half, uint32_t> const& idx,
                   cagra::search_params const& p) {
      auto found = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, top_k);
      auto dist  = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k);
      // The first call also loads the kernels: time the second.
      cagra::search(
        res, p, idx, raft::make_const_mdspan(queries.view()), found.view(), dist.view(), filter);
      raft::resource::sync_stream(res);
      auto started = std::chrono::steady_clock::now();
      cagra::search(
        res, p, idx, raft::make_const_mdspan(queries.view()), found.view(), dist.view(), filter);
      raft::resource::sync_stream(res);
      std::chrono::duration<double, std::milli> took = std::chrono::steady_clock::now() - started;
      std::cout << "  " << name << ": recall " << recall(to_host(res, found), truth) << ", "
                << took.count() << " ms for " << n_queries << " queries" << std::endl;
    };
    std::cout << "filter passing " << passing << " of " << rows.n_rows << " rows ("
              << 100 * pass_rate << "%), k " << top_k << std::endl;
    run("two-hop walk, graph index", graph, two_hop);
    run("FP8 scan, graph index (AUTO would choose it below scan_max_pass_rate)", graph, scan);
    run("FP8 scan, graph-free index", scan_only, scan);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  raft::device_resources res;
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  std::string mode = argc > 1 ? argv[1] : "demo";
  if (mode == "demo") {
    std::string out_dir = ".";
    auto rows           = synthetic_rows(res, 200000, 64);
    build(res, rows, out_dir);
    search(res, out_dir);
  } else if (mode == "build" && argc == 4) {
    build(res, read_fbin_as_half(argv[2]), argv[3]);
  } else if (mode == "search" && argc == 3) {
    search(res, argv[2]);
  } else {
    std::cerr << "usage: " << argv[0] << " [build DATA.fbin OUT_DIR | search OUT_DIR]" << std::endl;
    return 2;
  }
  return 0;
}
