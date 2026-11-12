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

// How a MULTI_CTA search walks the graph when a filter is given
type FilterExpansion int

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

// Fraction of the index's rows the filter removes. More than 0, less than 1.
// When 0 (the default), it is computed from the filter by each search that needs it,
// which reads the whole bitset and waits for the result.
func (p *SearchParams) SetFilteringRate(filtering_rate float32) (*SearchParams, error) {
	p.params.filtering_rate = C.float(filtering_rate)
	return p, nil
}

// How a filtered MULTI_CTA search walks the graph. With FilterExpansionPassingTwoHop,
// SearchAlgoAuto selects MULTI_CTA. Other algorithms ignore it.
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

// One bitset per query for a filtered MULTI_CTA search. No per-query bitsets when 0.
// Otherwise the device address of an array with one pointer per query, each to the 32-bit words
// of that query's own bitset over the index's rows (bit i % 32 of word i / 32 is set when row i
// passes), on the device being searched. A null pointer leaves that query unfiltered.
// The search must still be given a non-nil allowList. Its bitset is then not read, so queries
// with different filters can share one search call.
// The filtering rate must be given unless the filter expansion is FilterExpansionPassingTwoHop,
// and the algo must be SearchAlgoMultiCta or SearchAlgoAuto, which then selects it.
// The address is stored as given and used by every search with these params. SearchParams does
// not own the array or the bitsets: they must stay valid until the search's work on the stream
// has finished (see Resource.Sync).
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
