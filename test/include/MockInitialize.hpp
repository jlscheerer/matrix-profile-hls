/**
 * @file    MockInitialize.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   TODO: Add Brief Description
 */

#pragma once

namespace MatrixProfileTests {

    template<typename T>
    constexpr T AggregateInit();

    template<>
    constexpr double AggregateInit() { return -1e12; }

    template<>
    constexpr float AggregateInit() { return -1e12; }

    template<typename T>
    constexpr T IndexInit();

    template<>
    constexpr int IndexInit() { return -1; }

}