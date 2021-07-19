/**
 * @file    MatrixProfile.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Kernel Implementation Definition, Constant Definitions (n, m) and Type Aliases (data_t, index_t)
 */

#pragma once

#include "ArbitraryPrecisionFixed.hpp"
#include "AggregateTypeTraits.hpp"

// type alias for time series & resulting matrix profile
using data_t = double;

// type alias for the resulting matrix profile index
using index_t = int;

// length of the time series data
static constexpr index_t n = 8;

// subsequence length for the matrix profile
static constexpr index_t m = 4;

// assumption: t ≥ m
static constexpr index_t t = 4;

// length of the resulting matrix profile (index)
static constexpr index_t sublen = n - m + 1;

// "negative infinity" used to initialize aggregates
static const data_t aggregate_init = AggregateInit<data_t>::value;

// used to indicate an invalid/undetermined index
static const index_t index_init = IndexInit<index_t>::value;

static constexpr bool target_embedded = false;

struct ComputePack { data_t df, dg, inv; };

// "translate" the selected kernel implementation (type) to a readable string
#define KERNEL_IMPL_NAME "vStreamless"
#define KERNEL_IMPL_INDEX 0

#define DATA_TYPE_NAME "double"
#define INDEX_TYPE_NAME "int"
#define TARGET_TYPE_NAME "sw_emu"
