---
slug: api-reference/rust-api-cuvs-neighbors-cagra
---

# Neighbors Cagra Module

_Rust module: `cuvs::neighbors::cagra`_

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs`_

CAGRA: a graph-based approximate nearest neighbors algorithm with
state-of-the-art query throughput for both small and large batch sizes.

Build an [`Index`] from a dataset, then [`search`](Index::search) it with
device-resident queries and output buffers. Tensors are passed through the
`AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
module for the tensor model and `examples/cagra.rs` for a complete, runnable
example.

Parameter types ([`IndexParams`], [`SearchParams`], ...) use the [`bon`]
builder pattern: every setter is optional and unset values keep the cuVS C
library defaults. Values are validated when the builder's `build()` runs,
returning [`CagraError::Validation`] for out-of-range inputs.

## crate::dataset::\{CuvsDataset, Dataset, DatasetKind, DatasetView, PaddedDataset\}

```rust
pub use crate::dataset::{CuvsDataset, Dataset, DatasetKind, DatasetView, PaddedDataset};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:23`_

## crate::neighbors::filters::\{Bitset, Filter\}

```rust
pub use crate::neighbors::filters::{Bitset, Filter};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:24`_

## index::\{DeserializedIndex, Index\}

```rust
pub use index::{DeserializedIndex, Index};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:25`_

## params::\{IndexParams, SearchParams\}

```rust
pub use params::{IndexParams, SearchParams};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:26`_

## GraphBuildAlgo

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphBuildAlgo {
    /* variants omitted */
}
```

Algorithm for building the internal k-NN graph.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:34`_

## SearchAlgo

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchAlgo {
    /* variants omitted */
}
```

Search kernel implementation.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:74`_

## HashMode

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum HashMode {
    /* variants omitted */
}
```

Hash-table mode used during search.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:110`_

## FilterExpansion

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilterExpansion {
    /* variants omitted */
}
```

How a filtered CAGRA search traverses the graph.

[`FilterExpansion::PassingTwoHop`] is intended for filters that pass a small
fraction of the dataset, where [`FilterExpansion::ScoreAll`] computes most
distances for nodes that are filtered out. The search only moves between
nodes that pass the filter and are at most two hops apart in the graph, so
[`FilterExpansion::PassingTwoHop`] is effective when the rows that pass the
filter are close to each other in the graph. When fewer than roughly one
row in `graph_degree * graph_degree` passes the filter, the search cannot
move beyond its initial random seed nodes. When the rows that pass are
spread uniformly over the dataset at a rate not far above that, the search
converges slowly and needs a much larger `itopk_size` than
[`FilterExpansion::ScoreAll`] for the same recall. Use
[`FilterExpansion::ScoreAll`] or an exact search
([`brute_force`](crate::neighbors::brute_force) with a bitset filter) for
such filters.

[`FilterExpansion::PassingTwoHop`] is implemented by
[`SearchAlgo::MultiCta`] only. It selects [`SearchAlgo::MultiCta`] when
`algo` is [`SearchAlgo::Auto`], unless `persistent` is set. The other
algorithms ignore it and log a warning. `algo` defaults to
[`SearchAlgo::SingleCta`], so it must be set to [`SearchAlgo::MultiCta`]
(or [`SearchAlgo::Auto`]) for [`FilterExpansion::PassingTwoHop`] to take
effect. The value has no effect on a search without a filter.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:165`_

## CagraError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CagraError {
    /* variants omitted */
}
```

Error type for CAGRA operations.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:207`_
