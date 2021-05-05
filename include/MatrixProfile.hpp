#pragma once
#include <cstdlib>

// type alias for time series & resulting matrix profile
using data_t = double;

// type alias for the resulting matrix profile index
using index_t = int;

// length of the time series data
static constexpr size_t n = 8;

// subsequence length for the matrix profile
static constexpr size_t m = 4;

// length of the resulting matrix profile (index)
static constexpr size_t rs_len = n - m + 1;
