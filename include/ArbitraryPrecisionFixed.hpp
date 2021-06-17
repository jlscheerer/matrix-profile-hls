/**
 * @file    ArbitraryPrecisionFixed.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   TODO: Add Brief Description
 */

#pragma once

#include <ap_fixed.h>

// Arbitrary Precision Fixed-Point Data Type - Definitions
// For More Information visit https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/use_arbitrary_precision_data_type.html

// 16 bit Arbitrary Precision Fixed-Point Data Type
// 5 bits (incl. sign) for integral part; 11 bits for fractional part
// AP_RND_ZERO=Round to zero; AP_WRAP_SM=Sign magnitude wrap around
// [-16, 15] . (2048 / 2047)
using ap16_t = ap_fixed<16, 5, AP_RND_ZERO, AP_WRAP_SM>;

// 24 bit Arbitrary Precision Fixed-Point Data Type
// 8 bits (incl. sign) for integral part; 16 bits for fractional part
// AP_RND_ZERO=Round to zero; AP_WRAP_SM=Sign magnitude wrap around
// [-128, 127] . (65535 / 65536)
using ap24_t = ap_fixed<24, 8, AP_RND_ZERO, AP_WRAP_SM>;

// 32 bit Arbitrary Precision Fixed-Point Data Type
// 11 bits (incl. sign) for integral part; 21 bits for fractional part
// AP_RND_ZERO=Round to zero; AP_WRAP_SM=Sign magnitude wrap around
// [-1024, 1023] . (2097151 / 2097152)
using ap32_t = ap_fixed<32, 11, AP_RND_ZERO, AP_WRAP_SM>;

// 64 bit Arbitrary Precision Fixed-Point Data Type
// 14 bits (incl. sign) for integral part; 50 bits for fractional part
// AP_RND_ZERO=Round to zero; AP_WRAP_SM=Sign magnitude wrap around
// [-8192, 8191] . (1125899906842623.0 / 1125899906842624)
using ap64_t = ap_fixed<64, 14, AP_RND_ZERO, AP_WRAP_SM>;
