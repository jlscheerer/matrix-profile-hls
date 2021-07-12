/**
 * @file    HLSMathUtil.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Utiliy Functions related to HLS Math
 */

#pragma once

template<int W, int I>
ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM> sqrt(ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM> x) {
    return sqrt(x.to_double());
}