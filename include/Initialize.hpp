
#pragma once

template<typename data_t>
constexpr data_t AggregateInit();

template<>
constexpr double AggregateInit() { return -1e12; }

template<typename index_t>
constexpr index_t IndexInit();

template<>
constexpr int IndexInit() { return -1; }