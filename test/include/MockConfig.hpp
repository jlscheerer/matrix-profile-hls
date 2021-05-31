#pragma once

// length of the resulting matrix profile (index)
static constexpr size_t sublen = n - m + 1;

// "negative infinity" used to initialize aggregates
static constexpr data_t aggregate_init = -1e12;

// used to indicate an invalid/undetermined index
static constexpr index_t index_init = -1;

typedef struct {
    data_t value;
    index_t index;
} aggregate_t;

static constexpr aggregate_t aggregate_t_init{aggregate_init, index_init};