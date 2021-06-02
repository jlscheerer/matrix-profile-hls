
#pragma once

template<typename T>
static constexpr T AggregateInit();

template<>
static constexpr double AggregateInit() { return -1e12; }

template<typename T>
static constexpr index_t IndexInit();

template<>
static constexpr int IndexInit() { return -1; }