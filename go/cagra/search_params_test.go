package cagra

import (
	"testing"
)

func TestCreateSearchParams(t *testing.T) {
	params, err := CreateSearchParams()
	if err != nil {
		t.Fatalf("Failed to create SearchParams: %v", err)
	}
	defer params.Close()

	if params == nil {
		t.Fatal("CreateSearchParams returned nil params")
	}

	if params.params == nil {
		t.Fatal("SearchParams internal params are nil")
	}

	if float32(params.params.filtering_rate) != 0 {
		t.Errorf("Expected default filtering_rate 0, got %v", params.params.filtering_rate)
	}

	if uint32(params.params.filter_expansion) != 0 {
		t.Errorf("Expected default filter_expansion 0, got %d", params.params.filter_expansion)
	}

	if uintptr(params.params.query_bitsets) != 0 {
		t.Errorf("Expected default query_bitsets 0, got %#x", params.params.query_bitsets)
	}
}

func TestSearchParamsSetFilteringRate(t *testing.T) {
	params, err := CreateSearchParams()
	if err != nil {
		t.Fatalf("Failed to create SearchParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value float32
	}{
		{"Low rate", 0.1},
		{"Medium rate", 0.5},
		{"High rate", 0.99},
		{"Zero value (computed by the search)", 0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetFilteringRate(tc.value)
			if err != nil {
				t.Errorf("SetFilteringRate failed: %v", err)
			}
			if result != params {
				t.Error("SetFilteringRate should return the same params instance")
			}
			if float32(params.params.filtering_rate) != tc.value {
				t.Errorf("Expected filtering_rate %v, got %v",
					tc.value, params.params.filtering_rate)
			}
		})
	}
}

func TestSearchParamsSetFilterExpansion(t *testing.T) {
	params, err := CreateSearchParams()
	if err != nil {
		t.Fatalf("Failed to create SearchParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name          string
		expansion     FilterExpansion
		expected      uint32
		expectedError bool
	}{
		{"PASSING_TWO_HOP expansion", FilterExpansionPassingTwoHop, 1, false},
		{"Invalid expansion", FilterExpansion(999), 1, true},
		{"SCORE_ALL expansion", FilterExpansionScoreAll, 0, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetFilterExpansion(tc.expansion)

			if tc.expectedError {
				if err == nil {
					t.Error("Expected error for invalid filter_expansion, got none")
				}
			} else {
				if err != nil {
					t.Errorf("SetFilterExpansion failed: %v", err)
				}
				if result != params {
					t.Error("SetFilterExpansion should return the same params instance")
				}
			}
			// An invalid value leaves the previous one in place
			if uint32(params.params.filter_expansion) != tc.expected {
				t.Errorf("Expected filter_expansion %d, got %d",
					tc.expected, params.params.filter_expansion)
			}
		})
	}
}

func TestSearchParamsSetQueryBitsets(t *testing.T) {
	params, err := CreateSearchParams()
	if err != nil {
		t.Fatalf("Failed to create SearchParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value uintptr
	}{
		{"Device address", 0x7f1234560000},
		{"Zero value (no per-query bitsets)", 0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetQueryBitsets(tc.value)
			if err != nil {
				t.Errorf("SetQueryBitsets failed: %v", err)
			}
			if result != params {
				t.Error("SetQueryBitsets should return the same params instance")
			}
			if uintptr(params.params.query_bitsets) != tc.value {
				t.Errorf("Expected query_bitsets %#x, got %#x",
					tc.value, params.params.query_bitsets)
			}
		})
	}
}
