/**
 * @file    MatrixProfileKernelStream1D.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Stream-1D]
 */

#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_math.h"
#include "hls_stream.h"

using hls::stream;
constexpr size_t stream_d = 2;

void MemoryToStream(const data_t *T, stream<data_t, stream_d> &QT, stream<data_t, stream_d> &df_i,
        stream<data_t, stream_d> &df_j, stream<data_t, stream_d> &dg_i, stream<data_t, stream_d> &dg_j, stream<data_t, stream_d> &inv_i, stream<data_t, stream_d> &inv_j,
        stream<data_t, stream_d> &rowAggregate, stream<index_t, stream_d> &rowAggregateIndex, stream<data_t, stream_d> &columnAggregate,
        stream<index_t, stream_d> &columnAggregateIndex) {
    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];

    // the first m T values, required for convolution
    data_t Ti_m[m];

    // TODO: Move mean out to unroll
    data_t mean = 0;
    for (size_t i = 0; i < m; i++) {
        data_t T_i = T[i];
        mean += T_i;
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }
    mean /= m;
    data_t mu0 = mean;

    // TODO: Unroll this Loop
    data_t inv_sum = 0, qt_sum = 0;
    for (size_t k = 0; k < m; k++) {
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }
    data_t inv = 1 / sqrt(inv_sum);

    QT.write(qt_sum);
    df_i.write(0); df_j.write(0);
    dg_i.write(0); dg_j.write(0);
    inv_i.write(inv); inv_j.write(inv);

    rowAggregate.write(aggregate_init); rowAggregateIndex.write(index_init);
    columnAggregate.write(aggregate_init); columnAggregateIndex.write(index_init);

    data_t prev_mean;
    for (size_t i = m; i < n; ++i) {
        data_t T_i = T[i];
        data_t T_r = T_m[0];

        // set and update mean
        // TODO: Recompute Mean to achieve II=1
        prev_mean = mean;
        // mu[i - m + 1] = mean;
        mean = mean + (T_i - T_r) / m;

        inv_sum = 0; qt_sum = 0;
        // TODO: needs to be unrolled
        for (size_t k = 1; k < m; k++) {
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }
        // perform last element of the loop separately (this requires the newly read value)
        qt_sum += (T_i - mean) * (Ti_m[m - 1] - mu0);
        // QT[i - m + 1] = qt_sum;
        QT.write(qt_sum);

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        // df[i - m + 1] = (T_i - T_r) / 2;
        data_t df = (T_i - T_r) / 2;
        df_i.write(df); df_j.write(df);

        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        // dg[i - m + 1] = (T_i - mean) + (T_r - prev_mean);
        data_t dg = (T_i - mean) + (T_r - prev_mean);
        dg_i.write(dg); dg_j.write(dg);

        inv_sum += (T_i - mean) * (T_i - mean);
        inv = 1 / sqrt(inv_sum);

        // inv[i - m + 1] = 1 / Math.sqrt(inv_sum);
        inv_i.write(inv); inv_j.write(inv);

        rowAggregate.write(aggregate_init); rowAggregateIndex.write(index_init);
        columnAggregate.write(aggregate_init); columnAggregateIndex.write(index_init);

        // shift all values in T_m back
        // TODO: Unroll
        for (size_t k = 0; k < m - 1; k++) {
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = T_i;
    }
}

constexpr bool ExclusionZone(size_t i, size_t j) {
    // Exclusion Zone <==> i - m/4 <= j <= i + m/4
    // 				  <==> j <= i + m/4 [i <= j, m > 0]
    return j <= i + m / 4;
}

// TODO: Can optimize exclusionZone either every result will be in the exclusionZone or none
void MatrixProfileComputationElement(size_t stage, stream<data_t, stream_d> &QT_in,
        stream<data_t, stream_d> &df_i_in, stream<data_t, stream_d> &df_j_in, stream<data_t, stream_d> &dg_i_in, stream<data_t, stream_d> &dg_j_in,
        stream<data_t, stream_d> &inv_i_in, stream<data_t, stream_d> &inv_j_in,

        stream<data_t, stream_d> &rowAggregate_in, stream<index_t, stream_d> &rowAggregateIndex_in, stream<data_t, stream_d> &columnAggregate_in,
        stream<index_t, stream_d> &columnAggregateIndex_in,

        stream<data_t, stream_d> &QT_out, stream<data_t, stream_d> &df_i_out, stream<data_t, stream_d> &df_j_out, stream<data_t, stream_d> &dg_i_out,
        stream<data_t, stream_d> &dg_j_out, stream<data_t, stream_d> &inv_i_out, stream<data_t, stream_d> &inv_j_out,

        stream<data_t, stream_d> &rowAggregate_out, stream<index_t, stream_d> &rowAggregateIndex_out, stream<data_t, stream_d> &columnAggregate_out,
        stream<index_t, stream_d> &columnAggregateIndex_out) {
    for (size_t i = 0; i < stage; ++i) {
        // forward column aggregate
        data_t columnAggregate = columnAggregate_in.read();
        index_t columnAggregateIndex = columnAggregateIndex_in.read();
        columnAggregate_out.write(columnAggregate);
        columnAggregateIndex_out.write(columnAggregateIndex);
    }

    // take care of the first iteration
    data_t QT = QT_in.read();
    data_t df_i = df_i_in.read(); data_t df_j = df_j_in.read();
    data_t dg_i = dg_i_in.read(); data_t dg_j = dg_j_in.read();
    data_t inv_i = inv_i_in.read(); data_t inv_j = inv_j_in.read();

    data_t rowAggregate = rowAggregate_in.read(); index_t rowAggregateIndex = rowAggregateIndex_in.read();
    data_t columnAggregate = columnAggregate_in.read(); index_t columnAggregateIndex = columnAggregateIndex_in.read();

    data_t P = QT * inv_i * inv_j;
    bool exclusionZone = ExclusionZone(0, stage);

    // pass on the column aggregate (first column)
    if (!exclusionZone && P >= columnAggregate) {
        // use our value
        columnAggregate_out.write(P);
        columnAggregateIndex_out.write(0); // remember "best" row for columns
    } else {
        // pass on previous aggregate
        columnAggregate_out.write(columnAggregate);
        columnAggregateIndex_out.write(columnAggregateIndex);
    }

    // previous row values hold for one iteration of the loop
    data_t pdf_i = df_i, pdg_i = dg_i, pinv_i = inv_i;
    data_t pP = P; bool pExclusionZone = exclusionZone;
    data_t pRowAggregate = rowAggregate; index_t pRowAggregateIndex = rowAggregateIndex;
    for (size_t i = 1; i < n - m + 1 - stage; ++i) {
        data_t pQT = QT_in.read();

        df_i = df_i_in.read(); df_j = df_j_in.read();
        dg_i = dg_i_in.read(); dg_j = dg_j_in.read();
        inv_i = inv_i_in.read(); inv_j = inv_j_in.read();

        rowAggregate = rowAggregate_in.read(); rowAggregateIndex = rowAggregateIndex_in.read();
        columnAggregate = columnAggregate_in.read(); columnAggregateIndex = columnAggregateIndex_in.read();

        QT_out.write(pQT);
        df_i_out.write(pdf_i); df_j_out.write(df_j);
        dg_i_out.write(pdg_i); dg_j_out.write(dg_j);
        inv_i_out.write(pinv_i); inv_j_out.write(inv_j);

        QT = QT + df_i * dg_j + df_j * dg_i;
        P = QT * inv_i * inv_j;
        exclusionZone = ExclusionZone(i, stage + i);

        // pass on the column aggregate
        if (!exclusionZone && P >= columnAggregate) {
            // use our value
            columnAggregate_out.write(P);
            columnAggregateIndex_out.write(i); // remember "best" row for columns
        } else {
            // pass on previous aggregate
            columnAggregate_out.write(columnAggregate);
            columnAggregateIndex_out.write(columnAggregateIndex);
        }

        // pass on the row aggregate
        if (!pExclusionZone && pP >= pRowAggregate) {
            // use our value
            rowAggregate_out.write(pP);
            rowAggregateIndex_out.write(i + stage - 1); // remember "best" column // for rows
        } else {
            // pass on previous aggregate
            rowAggregate_out.write(pRowAggregate);
            rowAggregateIndex_out.write(pRowAggregateIndex);
        }

        // Shift everything backward
        pdf_i = df_i; pdg_i = dg_i; pinv_i = inv_i;
        pP = P; pExclusionZone = exclusionZone;
        pRowAggregate = rowAggregate; pRowAggregateIndex = rowAggregateIndex;
    }

    // pass on the row aggregate (last row)
    if (!exclusionZone && pP >= pRowAggregate) {
        // use our value
        rowAggregate_out.write(pP);
        rowAggregateIndex_out.write(n - m); // remember "best" column for rows
    } else {
        // pass on previous aggregate
        rowAggregate_out.write(pRowAggregate);
        rowAggregateIndex_out.write(pRowAggregateIndex);
    }

    // forward row aggregates
    for (size_t i = 0; i < stage; ++i) {
        rowAggregate = rowAggregate_in.read();
        rowAggregateIndex = rowAggregateIndex_in.read();
        rowAggregate_out.write(rowAggregate);
        rowAggregateIndex_out.write(rowAggregateIndex);
    }
}

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void StreamToMemory(stream<data_t, stream_d> &rowAggregate, stream<index_t, stream_d> &rowAggregateIndex,
        stream<data_t, stream_d> &columnAggregate, stream<index_t, stream_d> &columnAggregateIndex, data_t *MP, index_t *MPI) {
    // TODO: Use local cache don't directly write to memory (might be better?) (this way we need to access 3 times!)
    for (size_t i = 0; i < n - m + 1; ++i){
        // TODO: If this remains move to constant
        MP[i] = 1e12; // i.e. "positive infinity"
        MPI[i] = -1;
    }

    // Just rows
    for (size_t i = 0; i < n - m + 1; ++i) {
        data_t rowAggr = rowAggregate.read();
        index_t rowAggrIndex = rowAggregateIndex.read();
        // TODO: Improve this (move out MP?, calculate PearsonCorrelation only once)
        data_t euclideanDistance = PearsonCorrelationToEuclideanDistance(rowAggr);
        if (euclideanDistance < MP[i]) {
            MP[i] = euclideanDistance;
            MPI[i] = rowAggrIndex;
        }
    }

    // Just columns
    for (size_t i = 0; i < n - m + 1; ++i) {
        data_t columnAggr = columnAggregate.read();
        index_t columnAggrIndex = columnAggregateIndex.read();
        // TODO: Improve this (move out MP?, calculate PearsonCorrelation only once)
        data_t euclideanDistance = PearsonCorrelationToEuclideanDistance(columnAggr);
        if (euclideanDistance < MP[i]) {
            MP[i] = euclideanDistance;
            MPI[i] = columnAggrIndex;
        }
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    constexpr size_t numStages = n - m + 1;

    // Streams for the Scatter Lane (i: rows, j: columns)
    stream<data_t, stream_d> QT[numStages + 1];
    stream<data_t, stream_d> df_i[numStages + 1], df_j[numStages + 1];
    stream<data_t, stream_d> dg_i[numStages + 1], dg_j[numStages + 1];
    stream<data_t, stream_d> inv_i[numStages + 1], inv_j[numStages + 1];

    // Streams for the Reduction Lane (Aggregates)
    stream<data_t, stream_d> rowAggregate[numStages + 1], rowAggregateIndex[numStages + 1];
    stream<data_t, stream_d> columnAggregate[numStages + 1], columnAggregateIndex[numStages + 1];

    MemoryToStream(T, QT[0], df_i[0], df_j[0], dg_i[0], dg_j[0], inv_i[0], inv_j[0], rowAggregate[0],
            rowAggregateIndex[0], columnAggregate[0], columnAggregateIndex[0]);

    for (size_t stage = 0; stage < numStages; ++stage) {
        MatrixProfileComputationElement(stage, QT[stage], df_i[stage], df_j[stage], dg_i[stage], dg_j[stage],
                inv_i[stage], inv_j[stage], rowAggregate[stage], rowAggregateIndex[stage], columnAggregate[stage],
                columnAggregateIndex[stage], QT[stage + 1], df_i[stage + 1], df_j[stage + 1], dg_i[stage + 1],
                dg_j[stage + 1], inv_i[stage + 1], inv_j[stage + 1], rowAggregate[stage + 1],
                rowAggregateIndex[stage + 1], columnAggregate[stage + 1], columnAggregateIndex[stage + 1]);
    }

    StreamToMemory(rowAggregate[numStages], rowAggregateIndex[numStages], columnAggregate[numStages],
            columnAggregateIndex[numStages], MP, MPI);
}
