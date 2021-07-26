/**
 * @file    DataPacks.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Definition of Data Packs (DataPack, ComputePack)
 */

#pragma once

// Structure containing all values required for Update
// Computation and Conversion to PearsonCorrelation
struct DataPack { 
    data_t df, dg, inv;
    DataPack() = default;
    DataPack(data_t df, data_t dg, data_t inv)
        : df(df), dg(dg), inv(inv) {}
    DataPack(data_t v)
        : df(v), dg(v), inv(v) {}
};

struct ComputePack {
    DataPack row;
    aggregate_t rowAggregate;
    DataPack column;
    aggregate_t columnAggregate;
    ComputePack() = default;
    ComputePack(DataPack row, aggregate_t rowAggregate, DataPack column, aggregate_t columnAggregate)
        : row(row), rowAggregate(rowAggregate), column(column), columnAggregate(columnAggregate) {}
};