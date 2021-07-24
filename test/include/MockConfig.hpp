/**
 * @file    MockConfig.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Configuration for Software Testing
 */

#pragma once

template<typename T>
static constexpr T AggregateInit();

template<>
static constexpr double AggregateInit() { return -1e12; }

template<>
static constexpr float AggregateInit() { return -1e12; }

template<typename T>
static constexpr T IndexInit();

template<>
static constexpr int IndexInit() { return -1; }

// "negative infinity" used to initialize aggregates
static constexpr data_t aggregate_init = AggregateInit<data_t>();

// used to indicate an invalid/undetermined index
static constexpr index_t index_init = IndexInit<index_t>();

 struct aggregate_t { 
    data_t value; index_t index; 
    aggregate_t() = default;
    aggregate_t(const data_t value, const index_t index)
        : value(value), index(index) {}
    bool operator<(const data_t other) const { return value < other; }
    bool operator>(const aggregate_t other) const { return value > other.value; }
};
