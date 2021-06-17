/**
 * @file    AggregateTypeTraits.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   TODO: Add Brief Description
 */

#pragma once

#include <ap_fixed.h>

template<typename T>
struct AggregateInit {};

template<>
struct AggregateInit<double> { 
    static constexpr double value = -1e12; 
};

template<>
struct AggregateInit<float> { 
    static constexpr float value = -1e12; 
};

template<int W, int I>
struct AggregateInit<ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM>> {
    static const ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM> value;
};

template<int W, int I>
const ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM> AggregateInit<ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM>>::value = -((1LL << (I-1)));

template<typename T>
struct IndexInit {};

template<>
struct IndexInit<int> {
    static constexpr int value = -1;
};