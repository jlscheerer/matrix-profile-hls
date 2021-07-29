/**
 * @file    Stream.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Template alias for hls_stream class
 */

#pragma once

#if !defined(TEST_MOCK_SW)
    #include "hls_stream.h"
    using hls::stream;
#endif

static constexpr unsigned kStreamDepth = 3;

template<typename T>
using Stream = stream<T, kStreamDepth>;
