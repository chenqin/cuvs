# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
import weakref

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.common import make_device_padded_dataset
from cuvs.neighbors import cagra, filters, ivf_pq
from cuvs.tests.ann_utils import (
    calc_recall,
    generate_data,
    run_filtered_search_test,
)


def run_cagra_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    intermediate_graph_degree=128,
    graph_degree=64,
    build_algo="ivf_pq",
    array_type="device",
    compare=True,
    inplace=True,
    test_extend=False,
    search_params={},
    serialize=False,
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product" or metric == "cosine":
        if dtype in [np.int8, np.uint8]:
            pytest.skip("skip normalization for int8/uint8 data")
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
    )

    if test_extend:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        extend_params = cagra.ExtendParams()
        if array_type == "device":
            dataset_1_device = device_ndarray(dataset_1)

            index = cagra.build(build_params, dataset_1_device)
            new_start_row = dataset_1.shape[0]
            extended_dataset_owner = make_device_padded_dataset(
                device_ndarray(np.concatenate((dataset_1, dataset_2), axis=0))
            )
            index = cagra.extend(
                extend_params,
                index,
                extended_dataset_owner,
                new_start_row,
            )
        else:
            pytest.skip(
                "extend test path requires explicit padded dataset view"
            )
    else:
        if array_type == "device":
            index = cagra.build(build_params, dataset_device)
        else:
            index = cagra.build(build_params, dataset)

    if not compare:
        return

    if serialize:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_filename = f.name
        cagra.save(temp_filename, index)
        index = cagra.Index()
        out_dataset = cagra.Dataset()
        cagra.load(
            index,
            temp_filename,
            out_dataset=out_dataset,
        )

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = cagra.SearchParams(**search_params)

    ret_output = cagra.search(
        search_params,
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )

    if not inplace:
        out_dist_device, out_idx_device = ret_output

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

    # Calculate reference values with sklearn
    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "inner_product": "cosine",
        "euclidean": "euclidean",
        "cosine": "cosine",
    }[metric]
    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.7

    # test that we can get the cagra graph from the index
    graph = index.graph
    assert graph.shape == (n_rows, graph_degree)

    # make sure we can convert the graph to cupy, and access it
    cp_graph = cp.array(graph)
    assert cp_graph.shape == (n_rows, graph_degree)

    # make sure we can get the dataset from the cagra index
    dataset_from_index = index.dataset
    logical_dim = dataset.shape[1]
    dataset_from_index_host = dataset_from_index.copy_to_host()
    # CAGRA may store padded rows internally; compare only logical columns.
    dataset_from_index_host_logical = dataset_from_index_host[:, :logical_dim]
    assert np.allclose(dataset, dataset_from_index_host_logical)

    # make sure we can reconstruct the index from the graph
    # Note that we can't actually use the dataset from the index itself
    # - since that is a strided matrix (and we expect non-strided inputs
    # in the C++ cagra::build api), so we are using the host version
    # which will have been copied into a non-strided layout
    reloaded_dataset_device = device_ndarray(dataset_from_index_host_logical)
    reloaded_index = cagra.from_graph(
        graph, reloaded_dataset_device, metric=metric
    )
    reloaded_padded_dataset = make_device_padded_dataset(
        reloaded_dataset_device
    )
    cagra.update_dataset(reloaded_index, reloaded_padded_dataset)

    dist_device, idx_device = cagra.search(
        search_params, reloaded_index, queries_device, k
    )
    recall = calc_recall(idx_device.copy_to_host(), skl_idx)
    assert recall > 0.9


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("array_type", ["device"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product", "cosine"])
@pytest.mark.parametrize("serialize", [True, False])
def test_cagra_dataset_dtype_host_device(
    dtype, array_type, inplace, build_algo, metric, serialize
):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_build_search_test(
        dtype=dtype,
        inplace=inplace,
        array_type=array_type,
        build_algo=build_algo,
        metric=metric,
        serialize=serialize,
    )


@pytest.mark.parametrize("from_host", [True, False])
@pytest.mark.parametrize("n_rows", [1024, 2048])
@pytest.mark.parametrize("n_cols", [32, 50])
@pytest.mark.parametrize("n_queries", [32, 64])
@pytest.mark.parametrize("k", [5, 10])
def test_cagra_build_from_dataset_handle(
    from_host, n_rows, n_cols, n_queries, k
):
    dataset = generate_data((n_rows, n_cols), np.float32)
    padded = make_device_padded_dataset(
        dataset if from_host else device_ndarray(dataset)
    )
    assert isinstance(padded, cagra.Dataset)
    assert padded.layout == "padded"
    assert padded.memory_type == "device"

    index = cagra.build(cagra.IndexParams(), padded)
    assert index.trained

    queries = device_ndarray(generate_data((n_queries, n_cols), np.float32))
    distances, neighbors = cagra.search(
        cagra.SearchParams(), index, queries, k
    )

    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric="sqeuclidean"
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries.copy_to_host(), return_distance=False)
    assert calc_recall(neighbors.copy_to_host(), skl_idx) > 0.7
    assert distances.shape == (n_queries, k)


@pytest.mark.parametrize("sparsity", [0.2, 0.5, 0.7, 1.0])
def test_filtered_cagra(sparsity):
    run_filtered_search_test(cagra, sparsity)


def pack_bitset(mask):
    # Bit `i % 32` of word `i // 32` is set when row `i` passes the filter
    padded = np.zeros(-(-mask.size // 32) * 32, dtype=bool)
    padded[: mask.size] = mask
    return np.packbits(padded, bitorder="little").view(np.uint32)


def calc_filtered_recall(dataset, queries, neighbors, masks):
    # Checks that every returned row passes the mask of its query, and
    # calculates the recall against a brute-force search of the passing rows
    n_rows = dataset.shape[0]
    k = neighbors.shape[1]
    expected = np.zeros_like(neighbors)
    for i, mask in enumerate(masks):
        found = neighbors[i][neighbors[i] < n_rows]
        assert mask[found].all()
        passing = np.flatnonzero(mask)
        distances = ((dataset[passing] - queries[i]) ** 2).sum(axis=1)
        expected[i] = passing[np.argsort(distances)[:k]]
    return calc_recall(neighbors, expected)


def test_cagra_filter_expansion_two_hop(
    n_rows=10000, n_cols=10, n_queries=100, k=10, per_cluster=100
):
    # Clustered rows and a filter that passes whole clusters, so the rows
    # that pass the filter are close to each other
    n_clusters = n_rows // per_cluster
    centers = np.random.uniform(-1, 1, (n_clusters, n_cols))
    dataset = np.repeat(centers, per_cluster, axis=0)
    dataset += np.random.normal(0, 0.05, dataset.shape)
    dataset = dataset.astype(np.float32)
    queries = centers[np.random.randint(n_clusters, size=n_queries)]
    queries += np.random.normal(0, 0.2, queries.shape)
    queries = queries.astype(np.float32)
    mask = (np.arange(n_rows) // per_cluster) % 10 == 3

    index = cagra.build(cagra.IndexParams(), device_ndarray(dataset))

    search_params = cagra.SearchParams(
        algo="multi_cta", filter_expansion="passing_two_hop"
    )
    _, neighbors = cagra.search(
        search_params,
        index,
        device_ndarray(queries),
        k,
        filter=filters.from_bitset(device_ndarray(pack_bitset(mask))),
    )

    recall = calc_filtered_recall(
        dataset, queries, neighbors.copy_to_host(), [mask] * n_queries
    )
    assert recall > 0.9


@pytest.mark.parametrize(
    "filter_expansion, filtering_rate",
    [("score_all", 0.5), ("passing_two_hop", None)],
)
def test_cagra_query_bitsets(
    filter_expansion,
    filtering_rate,
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
):
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    # Even queries search one random half of the rows, odd queries the other
    half = np.random.random_sample(n_rows) < 0.5
    masks = [half, ~half]
    bitsets = [cp.asarray(pack_bitset(mask)) for mask in masks]
    query_bitsets = cp.array(
        [bitsets[i % 2].data.ptr for i in range(n_queries)], dtype=cp.uint64
    )

    index = cagra.build(cagra.IndexParams(), device_ndarray(dataset))

    search_params = cagra.SearchParams(
        algo="multi_cta",
        filtering_rate=filtering_rate,
        filter_expansion=filter_expansion,
        query_bitsets=query_bitsets,
    )
    # The bitset of the filter itself is not read
    _, neighbors = cagra.search(
        search_params,
        index,
        device_ndarray(queries),
        k,
        filter=filters.from_bitset(bitsets[0]),
    )

    recall = calc_filtered_recall(
        dataset,
        queries,
        neighbors.copy_to_host(),
        [masks[i % 2] for i in range(n_queries)],
    )
    assert recall > 0.9


def test_cagra_filter_search_params():
    search_params = cagra.SearchParams()
    assert search_params.filtering_rate == 0
    assert search_params.filter_expansion == "score_all"
    assert search_params.query_bitsets == 0

    query_bitsets = cp.zeros(4, dtype=cp.uint64)
    search_params = cagra.SearchParams(
        filtering_rate=0.9,
        filter_expansion="passing_two_hop",
        query_bitsets=query_bitsets,
    )
    assert np.isclose(search_params.filtering_rate, 0.9)
    assert search_params.filter_expansion == "passing_two_hop"
    assert search_params.query_bitsets == query_bitsets.data.ptr

    # The parameters keep the pointer array alive
    query_bitsets_ref = weakref.ref(query_bitsets)
    del query_bitsets
    assert query_bitsets_ref() is not None

    # A device address is taken as it is
    search_params = cagra.SearchParams(query_bitsets=0x1000)
    assert search_params.query_bitsets == 0x1000

    with pytest.raises(ValueError):
        cagra.SearchParams(filter_expansion="unknown")
    with pytest.raises(TypeError):
        cagra.SearchParams(query_bitsets=cp.zeros(4, dtype=cp.uint32))
    with pytest.raises(ValueError):
        cagra.SearchParams(query_bitsets=cp.zeros((2, 2), dtype=cp.uint64))
    with pytest.raises(TypeError):
        cagra.SearchParams(query_bitsets=np.zeros(4, dtype=np.uint64))


@pytest.mark.parametrize(
    "params",
    [
        {
            "dtype": np.int8,
            "intermediate_graph_degree": 64,
            "graph_degree": 32,
            "test_extend": False,
            "k": 1,
            "metric": "sqeuclidean",
            "build_algo": "ivf_pq",
        },
        {
            "dtype": np.float32,
            "intermediate_graph_degree": 32,
            "graph_degree": 16,
            "test_extend": True,
            "k": 5,
            "metric": "sqeuclidean",
            "build_algo": "ivf_pq",
        },
        {
            "dtype": np.float32,
            "intermediate_graph_degree": 128,
            "graph_degree": 32,
            "test_extend": False,
            "k": 10,
            "metric": "inner_product",
            "build_algo": "nn_descent",
        },
        {
            "dtype": np.float16,
            "intermediate_graph_degree": 64,
            "graph_degree": 32,
            "test_extend": True,
            "k": 10,
            "metric": "cosine",
            "build_algo": "ivf_pq",
        },
    ],
)
def test_cagra_index_params(params):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_build_search_test(
        dtype=params["dtype"],
        test_extend=params["test_extend"],
        k=params["k"],
        metric=params["metric"],
        graph_degree=params["graph_degree"],
        intermediate_graph_degree=params["intermediate_graph_degree"],
        compare=False,
        build_algo=params["build_algo"],
    )


@pytest.mark.parametrize("internal_dtype", [np.float32, np.float16, np.uint8])
def test_cagra_ivf_pq(
    internal_dtype,
    n_rows=1000,
    n_cols=30,
    n_queries=20,
    k=5,
    dtype=np.float16,
    metric="inner_product",
    intermediate_graph_degree=32,
    graph_degree=16,
    build_algo="ivf_pq",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    ivf_pq_params_build = ivf_pq.IndexParams(metric=metric, n_lists=10)
    ivf_pq_params_search = ivf_pq.SearchParams(
        n_probes=5,
        lut_dtype=internal_dtype,
        coarse_search_dtype=np.int8,
    )
    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
        ivf_pq_build_params=ivf_pq_params_build,
        ivf_pq_search_params=ivf_pq_params_search,
        refinement_rate=1.2,
    )
    cudadtype_to_np = {np.float32: 0, np.float16: 2, np.int8: 3, np.uint8: 8}

    assert (
        build_params.ivf_pq_search_params.lut_dtype
        == cudadtype_to_np[internal_dtype]
    )
    assert (
        build_params.ivf_pq_search_params.coarse_search_dtype
        == cudadtype_to_np[np.int8]
    )
    assert np.isclose(build_params.refinement_rate, 1.2)
    index = cagra.build(build_params, dataset_device)

    queries = generate_data((n_queries, n_cols), dtype)
    queries_device = device_ndarray(queries)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)
    out_idx_device = device_ndarray(out_idx)
    out_dist_device = device_ndarray(out_dist)

    cagra.search(
        cagra.SearchParams(),
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )
    out_idx = out_idx_device.copy_to_host()

    skl_idx = (
        NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine")
        .fit(dataset)
        .kneighbors(queries, return_distance=False)
    )

    recall = calc_recall(out_idx, skl_idx)

    assert recall > 0.9
