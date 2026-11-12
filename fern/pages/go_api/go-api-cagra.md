---
slug: api-reference/go-api-cagra
---

# Cagra Package

_Go package: `cagra`_

_Sources: `go/cagra`_

## Constants

### BuildAlgo Constants

```go
const (
IvfPq BuildAlgo = iota
NnDescent
AutoSelect
)
```

_Source: `go/cagra/index_params.go:18`_

### FilterExpansion Constants

```go
const (
// Every neighbor of an expanded node is scored and may be expanded in turn, whether or not it
// passes the filter; rejected nodes are dropped from the results. This is the default.
FilterExpansionScoreAll FilterExpansion = iota
// Only nodes that pass the filter are scored, kept and expanded. A rejected neighbor is not
// scored: its own neighbors are examined instead. The search starts from random nodes that
// pass the filter (num_random_samplings of them per start slot, the nearest kept).
// For filters that reject most of the index.
FilterExpansionPassingTwoHop
)
```

_Source: `go/cagra/search_params.go:37`_

### HashmapMode Constants

```go
const (
HashmapModeHash HashmapMode = iota
HashmapModeSmall
HashmapModeAuto
)
```

_Source: `go/cagra/search_params.go:28`_

### SearchAlgo Constants

```go
const (
SearchAlgoSingleCta SearchAlgo = iota
SearchAlgoMultiCta
SearchAlgoMultiKernel
SearchAlgoAuto
)
```

_Source: `go/cagra/search_params.go:19`_

## Types

### BuildAlgo

```go
type BuildAlgo int
```

_Source: `go/cagra/index_params.go:16`_

### CagraIndex

```go
type CagraIndex struct {
	// contains filtered or unexported fields
}
```

Cagra ANN Index

_Source: `go/cagra/cagra.go:14`_

### ExtendParams

```go
type ExtendParams struct {
	// contains filtered or unexported fields
}
```

Parameters to extend CAGRA Index

_Source: `go/cagra/extend_params.go:11`_

### FilterExpansion

```go
type FilterExpansion int
```

How a MULTI_CTA search walks the graph when a filter is given

_Source: `go/cagra/search_params.go:35`_

### HashmapMode

```go
type HashmapMode int
```

_Source: `go/cagra/search_params.go:26`_

### IndexParams

```go
type IndexParams struct {
	// contains filtered or unexported fields
}
```

_Source: `go/cagra/index_params.go:12`_

### PaddedDataset

```go
type PaddedDataset struct {
	// contains filtered or unexported fields
}
```

Owning padded dataset handle for explicit CAGRA dataset management.

_Source: `go/cagra/cagra.go:20`_

### PaddedDatasetHandle

```go
type PaddedDatasetHandle interface {
	datasetHandle() C.cuvsDataset_t
}
```

PaddedDatasetHandle is an owning padded dataset or non-owning padded dataset view.

_Source: `go/cagra/cagra.go:25`_

### PaddedDatasetView

```go
type PaddedDatasetView struct {
	// contains filtered or unexported fields
}
```

Non-owning padded dataset view handle.

_Source: `go/cagra/cagra.go:30`_

### SearchAlgo

```go
type SearchAlgo int
```

_Source: `go/cagra/search_params.go:17`_

### SearchParams

```go
type SearchParams struct {
	// contains filtered or unexported fields
}
```

Supplemental parameters to search CAGRA Index

_Source: `go/cagra/search_params.go:13`_

### StandardDatasetView

```go
type StandardDatasetView struct {
	// contains filtered or unexported fields
}
```

Non-owning standard dataset view handle.

_Source: `go/cagra/cagra.go:35`_

## Functions

### BuildIndex

```go
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *CagraIndex) error
```

Builds a new Index from the dataset for efficient search.

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters for building the index
* `dataset` - A row-major Tensor on either the host or device to index
* `index` - CagraIndex to build

_Source: `go/cagra/cagra.go:226`_

### CreateExtendParams

```go
func CreateExtendParams() (*ExtendParams, error)
```

Creates a new ExtendParams

_Source: `go/cagra/extend_params.go:16`_

### CreateIndex

```go
func CreateIndex() (*CagraIndex, error)
```

Creates a new empty Cagra Index

_Source: `go/cagra/cagra.go:208`_

### CreateIndexParams

```go
func CreateIndexParams() (*IndexParams, error)
```

Creates a new IndexParams

_Source: `go/cagra/index_params.go:31`_

### CreateSearchParams

```go
func CreateSearchParams() (*SearchParams, error)
```

Creates a new SearchParams

_Source: `go/cagra/search_params.go:49`_

### ExtendIndex

```go
func ExtendIndex(Resources cuvs.Resource, params *ExtendParams, extended_dataset PaddedDatasetHandle, newStartRow int64, index *CagraIndex) error
```

Extends the index with a caller-owned pre-concatenated padded dataset.

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters for extending the index
* `extended_dataset` - Caller-owned padded dataset already containing old \|\| new rows
* `newStartRow` - Row index where the additional vectors begin (must equal current index size)
* `index` - CagraIndex to extend

_Source: `go/cagra/cagra.go:277`_

### MakePaddedDataset

```go
func MakePaddedDataset[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDataset, error)
```

MakePaddedDataset creates an owning padded dataset from a tensor.
Memory residency is inferred from the tensor device type.

_Source: `go/cagra/cagra.go:84`_

### MakePaddedDatasetView

```go
func MakePaddedDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDatasetView, error)
```

MakePaddedDatasetView creates a non-owning padded dataset view from a tensor.
Memory residency is inferred from the tensor.

_Source: `go/cagra/cagra.go:109`_

### MakeStandardDatasetView

```go
func MakeStandardDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*StandardDatasetView, error)
```

MakeStandardDatasetView creates a non-owning standard dataset view from a tensor.
Memory residency is inferred from the tensor.

_Source: `go/cagra/cagra.go:159`_

### SearchIndex

```go
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *CagraIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[uint32], distances *cuvs.Tensor[T], allowList []uint32) error
```

Perform a Approximate Nearest Neighbors search on the Index

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters to use in searching the index
* `queries` - A tensor in device memory to query for
* `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
* `distances` - Tensor in device memory that receives the distances of the nearest neighbors
* `allowList` - List of indices to allow in the search, if nil, no filtering is applied

_Source: `go/cagra/cagra.go:317`_

### UpdateDataset

```go
func UpdateDataset(Resources cuvs.Resource, paddedDataset PaddedDatasetHandle, index *CagraIndex) error
```

UpdateDataset updates any CAGRA index layout with a caller-provided padded
dataset or view and leaves the same handle search-ready.

_Source: `go/cagra/cagra.go:189`_

## Methods

### CagraIndex.Close

```go
func (index *CagraIndex) Close() error
```

Destroys the Cagra Index

_Source: `go/cagra/cagra.go:299`_

### ExtendParams.Close

```go
func (p *ExtendParams) Close() error
```

_Source: `go/cagra/extend_params.go:40`_

### ExtendParams.SetMaxChunkSize

```go
func (p *ExtendParams) SetMaxChunkSize(max_chunk_size uint32) (*ExtendParams, error)
```

The additional dataset is divided into chunks and added to the graph.
This is the knob to adjust the tradeoff between the recall and operation throughput.
Large chunk sizes can result in high throughput, but use more
working memory (O(max_chunk_size*degree^2)).
This can also degrade recall because no edges are added between the nodes in the same chunk.
Auto select when 0.

_Source: `go/cagra/extend_params.go:35`_

### IndexParams.Close

```go
func (p *IndexParams) Close() error
```

Destroys IndexParams

_Source: `go/cagra/index_params.go:77`_

### IndexParams.SetBuildAlgo

```go
func (p *IndexParams) SetBuildAlgo(build_algo BuildAlgo) (*IndexParams, error)
```

ANN algorithm to build knn graph

_Source: `go/cagra/index_params.go:58`_

### IndexParams.SetGraphDegree

```go
func (p *IndexParams) SetGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error)
```

Degree of output graph

_Source: `go/cagra/index_params.go:51`_

### IndexParams.SetIntermediateGraphDegree

```go
func (p *IndexParams) SetIntermediateGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error)
```

Degree of input graph for pruning

_Source: `go/cagra/index_params.go:45`_

### IndexParams.SetNNDescentNiter

```go
func (p *IndexParams) SetNNDescentNiter(nn_descent_niter uint32) (*IndexParams, error)
```

Number of iterations to run if building with NN_DESCENT

_Source: `go/cagra/index_params.go:70`_

### PaddedDataset.Close

```go
func (dataset *PaddedDataset) Close() error
```

Destroys an owning padded dataset handle.

_Source: `go/cagra/cagra.go:132`_

### PaddedDatasetView.Close

```go
func (view *PaddedDatasetView) Close() error
```

Destroys a padded dataset view handle.

_Source: `go/cagra/cagra.go:145`_

### SearchParams.Close

```go
func (p *SearchParams) Close() error
```

Destroys SearchParams

_Source: `go/cagra/search_params.go:214`_

### SearchParams.SetAlgo

```go
func (p *SearchParams) SetAlgo(algo SearchAlgo) (*SearchParams, error)
```

Which search implementation to use.

_Source: `go/cagra/search_params.go:81`_

### SearchParams.SetFilterExpansion

```go
func (p *SearchParams) SetFilterExpansion(filter_expansion FilterExpansion) (*SearchParams, error)
```

How a filtered MULTI_CTA search walks the graph. With FilterExpansionPassingTwoHop,
SearchAlgoAuto selects MULTI_CTA. Other algorithms ignore it.

_Source: `go/cagra/search_params.go:180`_

### SearchParams.SetFilteringRate

```go
func (p *SearchParams) SetFilteringRate(filtering_rate float32) (*SearchParams, error)
```

Fraction of the index's rows the filter removes. More than 0, less than 1.
When 0 (the default), it is computed from the filter by each search that needs it,
which reads the whole bitset and waits for the result.

_Source: `go/cagra/search_params.go:173`_

### SearchParams.SetHashmapMaxFillRate

```go
func (p *SearchParams) SetHashmapMaxFillRate(hashmap_max_fill_rate float32) (*SearchParams, error)
```

Upper limit of hashmap fill rate. More than 0.1, less than 0.9.

_Source: `go/cagra/search_params.go:153`_

### SearchParams.SetHashmapMinBitlen

```go
func (p *SearchParams) SetHashmapMinBitlen(hashmap_min_bitlen uintptr) (*SearchParams, error)
```

Lower limit of hashmap bit length. More than 8.

_Source: `go/cagra/search_params.go:147`_

### SearchParams.SetHashmapMode

```go
func (p *SearchParams) SetHashmapMode(hashmap_mode HashmapMode) (*SearchParams, error)
```

Hashmap type. Auto selection when AUTO.

_Source: `go/cagra/search_params.go:127`_

### SearchParams.SetItopkSize

```go
func (p *SearchParams) SetItopkSize(itopk_size uintptr) (*SearchParams, error)
```

Number of intermediate search results retained during the search.
This is the main knob to adjust trade off between accuracy and search speed.
Higher values improve the search accuracy

_Source: `go/cagra/search_params.go:69`_

### SearchParams.SetMaxIterations

```go
func (p *SearchParams) SetMaxIterations(max_iterations uintptr) (*SearchParams, error)
```

Upper limit of search iterations. Auto select when 0.

_Source: `go/cagra/search_params.go:75`_

### SearchParams.SetMaxQueries

```go
func (p *SearchParams) SetMaxQueries(max_queries uintptr) (*SearchParams, error)
```

Maximum number of queries to search at the same time (batch size). Auto select when 0

_Source: `go/cagra/search_params.go:61`_

### SearchParams.SetMinIterations

```go
func (p *SearchParams) SetMinIterations(min_iterations uintptr) (*SearchParams, error)
```

Lower limit of search iterations.

_Source: `go/cagra/search_params.go:109`_

### SearchParams.SetNumRandomSamplings

```go
func (p *SearchParams) SetNumRandomSamplings(num_random_samplings uint32) (*SearchParams, error)
```

Number of iterations of initial random seed node selection. 1 or more.

_Source: `go/cagra/search_params.go:159`_

### SearchParams.SetQueryBitsets

```go
func (p *SearchParams) SetQueryBitsets(query_bitsets uintptr) (*SearchParams, error)
```

One bitset per query for a filtered MULTI_CTA search. No per-query bitsets when 0.
Otherwise the device address of an array with one pointer per query, each to the 32-bit words
of that query's own bitset over the index's rows (bit i % 32 of word i / 32 is set when row i
passes), on the device being searched. A null pointer leaves that query unfiltered.
The search must still be given a non-nil allowList. Its bitset is then not read, so queries
with different filters can share one search call.
The filtering rate must be given unless the filter expansion is FilterExpansionPassingTwoHop,
and the algo must be SearchAlgoMultiCta or SearchAlgoAuto, which then selects it.
The address is stored as given and used by every search with these params. SearchParams does
not own the array or the bitsets: they must stay valid until the search's work on the stream
has finished (see Resource.Sync).

_Source: `go/cagra/search_params.go:208`_

### SearchParams.SetRandXorMask

```go
func (p *SearchParams) SetRandXorMask(rand_xor_mask uint64) (*SearchParams, error)
```

Bit mask used for initial random seed node selection.

_Source: `go/cagra/search_params.go:165`_

### SearchParams.SetSearchWidth

```go
func (p *SearchParams) SetSearchWidth(search_width uintptr) (*SearchParams, error)
```

How many nodes to search at once. Auto select when 0.

_Source: `go/cagra/search_params.go:115`_

### SearchParams.SetTeamSize

```go
func (p *SearchParams) SetTeamSize(team_size uintptr) (*SearchParams, error)
```

Number of threads used to calculate a single distance. 4, 8, 16, or 32.

_Source: `go/cagra/search_params.go:103`_

### SearchParams.SetThreadBlockSize

```go
func (p *SearchParams) SetThreadBlockSize(thread_block_size uintptr) (*SearchParams, error)
```

Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.

_Source: `go/cagra/search_params.go:121`_

### StandardDatasetView.Close

```go
func (view *StandardDatasetView) Close() error
```

Destroys a standard dataset view handle.

_Source: `go/cagra/cagra.go:175`_
