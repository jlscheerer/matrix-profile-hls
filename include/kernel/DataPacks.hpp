/**
 * @file    DataPacks.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Definition of Data Packs (DataPack, ComputePack)
 */

#pragma once

struct ScatterPack { 
    data_t QT, df, dg, inv;
    ScatterPack() = default;
    ScatterPack(data_t QT, data_t df, data_t dg, data_t inv)
        : QT(QT), df(df), dg(dg), inv(inv) {}
    ScatterPack(data_t v)
        : QT(v), df(v), dg(v), inv(v) {}
};

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
    aggregate_t aggregate;
    data_t QTforward;
    ComputePack() = default;
    ComputePack(DataPack row, aggregate_t aggregate, data_t QTforward)
        : row(row), aggregate(aggregate), QTforward(QTforward) {}
};