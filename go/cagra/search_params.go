package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"

	cuvs "github.com/nvidia/cuvs/go"
)

// Supplemental parameters to search CAGRA Index
type SearchParams struct {
	params C.cuvsCagraSearchParams_t
}

type SearchAlgo int

const (
	SearchAlgoSingleCta SearchAlgo = iota
	SearchAlgoMultiCta
	SearchAlgoMultiKernel
	SearchAlgoAuto
)

type HashmapMode int

const (
	HashmapModeHash HashmapMode = iota
	HashmapModeSmall
	HashmapModeAuto
)

// How a filtered CAGRA search traverses the graph
//
// FilterExpansionPassingTwoHop is intended for filters that pass a small fraction of the dataset,
// where FilterExpansionScoreAll computes most distances for nodes that are filtered out. The
// search only moves between nodes that pass the filter and are at most two hops apart in the
// graph, so FilterExpansionPassingTwoHop is effective when the rows that pass the filter are close
// to each other in the graph. When fewer than roughly one row in graph_degree * graph_degree
// passes the filter, the search cannot move beyond its initial random seed nodes. When the rows
// that pass are spread uniformly over the dataset at a rate not far above that, the search
// converges slowly and needs a much larger itopk_size than FilterExpansionScoreAll for the same
// recall. Use FilterExpansionScoreAll or an exact search for such filters.
//
// Only the MULTI_CTA search algorithm implements FilterExpansionPassingTwoHop.
type FilterExpansion int

const (
	// The distance is computed for every neighbor of a parent node, whether or not it passes the
	// filter, and a filtered-out node can be selected as a parent node. Filtered-out nodes are
	// removed from the search results. The MULTI_CTA algorithm increases itopk_size according to
	// filtering_rate to compensate for the filtered-out nodes in the itopk list.
	// This is the default.
	FilterExpansionScoreAll FilterExpansion = iota
	// The distance is computed only for nodes that pass the filter, and only these nodes enter the
	// itopk list and are selected as parent nodes. The child nodes of a parent node are its
	// neighbors that pass the filter and, for each neighbor that is filtered out (a bridge), the
	// neighbors of the bridge that pass the filter, up to the child capacity of the result buffer
	// (about graph_degree nodes). The initial random seed nodes are random nodes that pass the
	// filter, each the nearest of up to num_random_samplings such nodes. itopk_size is not increased
	// according to filtering_rate, and the filtering rate of the filter is not computed.
	FilterExpansionPassingTwoHop
)

// Creates a new SearchParams
func CreateSearchParams() (*SearchParams, error) {
	var params C.cuvsCagraSearchParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

// Maximum number of queries to search at the same time (batch size). Auto select when 0
func (p *SearchParams) SetMaxQueries(max_queries uintptr) (*SearchParams, error) {
	p.params.max_queries = C.size_t(max_queries)
	return p, nil
}

// Number of intermediate search results retained during the search.
// This is the main knob to adjust trade off between accuracy and search speed.
// Higher values improve the search accuracy
func (p *SearchParams) SetItopkSize(itopk_size uintptr) (*SearchParams, error) {
	p.params.itopk_size = C.size_t(itopk_size)
	return p, nil
}

// Upper limit of search iterations. Auto select when 0.
func (p *SearchParams) SetMaxIterations(max_iterations uintptr) (*SearchParams, error) {
	p.params.max_iterations = C.size_t(max_iterations)
	return p, nil
}

// Which search implementation to use.
func (p *SearchParams) SetAlgo(algo SearchAlgo) (*SearchParams, error) {
	CAlgo := C.SINGLE_CTA

	switch algo {
	case SearchAlgoSingleCta:
		CAlgo = C.SINGLE_CTA
	case SearchAlgoMultiCta:
		CAlgo = C.MULTI_CTA
	case SearchAlgoMultiKernel:
		CAlgo = C.MULTI_KERNEL
	case SearchAlgoAuto:
		CAlgo = C.AUTO
	default:
		return nil, errors.New("unsupported algo")
	}

	p.params.algo = uint32(CAlgo)

	return p, nil
}

// Number of threads used to calculate a single distance. 4, 8, 16, or 32.
func (p *SearchParams) SetTeamSize(team_size uintptr) (*SearchParams, error) {
	p.params.team_size = C.size_t(team_size)
	return p, nil
}

// Lower limit of search iterations.
func (p *SearchParams) SetMinIterations(min_iterations uintptr) (*SearchParams, error) {
	p.params.min_iterations = C.size_t(min_iterations)
	return p, nil
}

// How many nodes to search at once. Auto select when 0.
func (p *SearchParams) SetSearchWidth(search_width uintptr) (*SearchParams, error) {
	p.params.search_width = C.size_t(search_width)
	return p, nil
}

// Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.
func (p *SearchParams) SetThreadBlockSize(thread_block_size uintptr) (*SearchParams, error) {
	p.params.thread_block_size = C.size_t(thread_block_size)
	return p, nil
}

// Hashmap type. Auto selection when AUTO.
func (p *SearchParams) SetHashmapMode(hashmap_mode HashmapMode) (*SearchParams, error) {
	CHashMode := C.AUTO_HASH

	switch hashmap_mode {
	case HashmapModeHash:
		CHashMode = C.HASH
	case HashmapModeSmall:
		CHashMode = C.SMALL
	case HashmapModeAuto:
		CHashMode = C.AUTO_HASH
	default:
		return nil, errors.New("unsupported hashmap_mode")
	}

	p.params.hashmap_mode = uint32(CHashMode)

	return p, nil
}

// Lower limit of hashmap bit length. More than 8.
func (p *SearchParams) SetHashmapMinBitlen(hashmap_min_bitlen uintptr) (*SearchParams, error) {
	p.params.hashmap_min_bitlen = C.size_t(hashmap_min_bitlen)
	return p, nil
}

// Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
func (p *SearchParams) SetHashmapMaxFillRate(hashmap_max_fill_rate float32) (*SearchParams, error) {
	p.params.hashmap_max_fill_rate = C.float(hashmap_max_fill_rate)
	return p, nil
}

// Number of iterations of initial random seed node selection. 1 or more.
func (p *SearchParams) SetNumRandomSamplings(num_random_samplings uint32) (*SearchParams, error) {
	p.params.num_random_samplings = C.uint32_t(num_random_samplings)
	return p, nil
}

// Bit mask used for initial random seed node selection.
func (p *SearchParams) SetRandXorMask(rand_xor_mask uint64) (*SearchParams, error) {
	p.params.rand_xor_mask = C.uint64_t(rand_xor_mask)
	return p, nil
}

// A parameter indicating the rate of nodes to be filtered-out, when filtering is used.
//
// The value must be greater than 0.0 and less than 1.0. When set to 0 (default), the filtering
// rate is automatically calculated: each filtered search call then counts the bits of the
// filter's bitset and waits for the result, unless the filter expansion is
// FilterExpansionPassingTwoHop, which does not use the rate.
func (p *SearchParams) SetFilteringRate(filtering_rate float32) (*SearchParams, error) {
	p.params.filtering_rate = C.float(filtering_rate)
	return p, nil
}

// How a filtered search traverses the graph (see FilterExpansion).
// Default: FilterExpansionScoreAll.
//
// FilterExpansionPassingTwoHop is implemented by the MULTI_CTA algorithm only. It selects
// SearchAlgoMultiCta when the algo is SearchAlgoAuto. The other algorithms ignore it and log a
// warning. CreateSearchParams leaves the algo at SearchAlgoSingleCta, so the algo must be set to
// SearchAlgoMultiCta (or SearchAlgoAuto) for FilterExpansionPassingTwoHop to take effect. The
// value has no effect on a search without a filter.
func (p *SearchParams) SetFilterExpansion(filter_expansion FilterExpansion) (*SearchParams, error) {
	CFilterExpansion := C.CUVS_CAGRA_FILTER_SCORE_ALL

	switch filter_expansion {
	case FilterExpansionScoreAll:
		CFilterExpansion = C.CUVS_CAGRA_FILTER_SCORE_ALL
	case FilterExpansionPassingTwoHop:
		CFilterExpansion = C.CUVS_CAGRA_FILTER_PASSING_TWO_HOP
	default:
		return nil, errors.New("unsupported filter_expansion")
	}

	p.params.filter_expansion = uint32(CFilterExpansion)

	return p, nil
}

// Per-query filter bitsets for MULTI_CTA search. Not used when 0 (default).
//
// When not 0, this is the device address of an array with one pointer per query of the search
// call. Each pointer is the device address of the 32-bit words of the bitset for that query, with
// one bit per row of the index: row i passes the filter when bit i % 32 of word i / 32 is set. A
// null pointer leaves that query unfiltered. The bitset of the allowList passed to the search is
// not read, so queries with different filters can share one search call.
//
// The search must be given a non-nil allowList. The algo must be SearchAlgoMultiCta or
// SearchAlgoAuto (CreateSearchParams leaves it at SearchAlgoSingleCta); SearchAlgoAuto selects
// SearchAlgoMultiCta, and the search fails with any other algorithm. With
// FilterExpansionScoreAll, the filtering rate must be greater than 0, because it is not
// calculated from per-query bitsets. FilterExpansionPassingTwoHop does not use it.
//
// The address is stored as given and used by every search with these params. SearchParams does
// not own the array or the bitsets: they must reside on the device that the search runs on and
// must remain valid until the work that the search submits to the CUDA stream has completed (see
// Resource.Sync).
func (p *SearchParams) SetQueryBitsets(query_bitsets uintptr) (*SearchParams, error) {
	p.params.query_bitsets = C.uintptr_t(query_bitsets)
	return p, nil
}

// Destroys SearchParams
func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
