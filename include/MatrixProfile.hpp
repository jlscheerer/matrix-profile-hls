/**
 * @file    MatrixProfile.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Kernel Implementation Definition, Constant Definitions (n, m) and Type Aliases (data_t, index_t)
 */

#pragma once

#include <cstdlib>

#define KERNEL_STREAMLESS 1
#define KERNEL_STREAM1D   2
#define KERNEL_STREAM2D   3

// this macro defines the (type of) implementation to
// use for the matrix profile kernel
#define KERNEL_IMPLEMENTATION KERNEL_STREAMLESS

// type alias for time series & resulting matrix profile
using data_t = double;

// type alias for the resulting matrix profile index
using index_t = int;

// length of the time series data
static constexpr size_t n = 8;

// subsequence length for the matrix profile
static constexpr size_t m = 4;

// "tile-size" (only applicable for Stream2D)
// assumption: t ≥ m
constexpr size_t t = 4;

// length of the resulting matrix profile (index)
static constexpr size_t sublen = n - m + 1;

// "negative infinity" used to initialize aggregates
static constexpr data_t aggregate_init = -1e12;

// used to indicate an invalid/undetermined index
static constexpr index_t index_init = -1;

// "translate" the selected kernel implementation (type) to a readable string
#if KERNEL_IMPLEMENTATION == KERNEL_STREAMLESS
    #define KERNEL_IMPL_NAME "vStreamless"
#elif KERNEL_IMPLEMENTATION == KERNEL_STREAM1D
    #define KERNEL_IMPL_NAME "vStream-1D"
#elif KERNEL_IMPLEMENTATION == KERNEL_STREAM2D
    #define KERNEL_IMPL_NAME "vStream-2D"
#else
    #error Invalid Kernel Implementation!
#endif
