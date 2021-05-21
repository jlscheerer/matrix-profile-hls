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

void MemoryToStream(const data_t *T, stream<data_t, 5> &QT, stream<data_t, 5> &df_i, stream<data_t, 5> &df_j,
                    stream<data_t, 5> &dg_i, stream<data_t, 5> &dg_j, stream<data_t, 5> &inv_i, stream<data_t, 5> &inv_j,
			        stream<data_t, 5> &rowWiseAggregate, stream<data_t, 5> &columnWiseAggregate,
                    stream<index_t, 5> &rowWiseIndex, stream<index_t, 5> &columnWiseIndex) {
    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];

    // the first m T values, required for convolution
    data_t Ti_m [m];

    data_t mean = 0, inv_sum = 0, qt_sum = 0;

    // initialize moving mean
    for(int i = 0; i < m; ++i) {
        data_t T_i = T[i];
        mean += T_i;
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }
    mean /= m;

    // calculate initial values
    data_t mu0 = mean;

    // initialize inv, qt calculations
    for (int k = 0; k < m; ++k) {
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }

    data_t inv = 1 / sqrt(inv_sum);

    inv_i.write(inv);
    inv_j.write(inv);

    QT.write(qt_sum);

    rowWiseAggregate.write(aggregate_init);
    rowWiseIndex.write(index_init);

    columnWiseAggregate.write(aggregate_init);
    columnWiseIndex.write(index_init);

    data_t prev_mean;
    for (int i = m; i < n; ++i) {
        data_t T_i = T[i];
        data_t T_r = T_m[0];

        // set and update mean
        prev_mean = mean;
        // calculate μ[i - m + 1]
        mean = mean + (T_i - T_r) / m;

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        // df[i - m + 1] = (T_i - T_r) / 2;
        data_t df = (T_i - T_r) / 2;
        df_i.write(df);
        df_j.write(df);

        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        // dg[i - m + 1] = (T_i - mean) + (T_r - prev_mean);
        data_t dg = (T_i - mean) + (T_r - prev_mean);
        dg_i.write(dg);
        dg_j.write(dg);

        inv_sum = 0;
        qt_sum = 0;

        for (int k = 1; k < m; ++k) {
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }
        // perform last element of the loop separately (this requires the newly read value)
        inv_sum += (T_i - mean) * (T_i - mean);

        // inv[i - m + 1] = 1 / sqrt(inv_sum);
        inv = 1 / sqrt(inv_sum);
        inv_i.write(inv);
        inv_j.write(inv);

        qt_sum += (T_i - mean) * (Ti_m[m - 1] - mu0);

        // QT[i - m + 1] = qt_sum;
        QT.write(qt_sum);

        rowWiseAggregate.write(aggregate_init);
        rowWiseIndex.write(index_init);

        columnWiseAggregate.write(aggregate_init);
        columnWiseIndex.write(index_init);

        // shift all values in T to the left
        for (int k = 0; k < m - 1; ++k)
            T_m[k] = T_m[k + 1];
        T_m[m - 1] = T_i;
    }
}

// D = sqrt(2 * m * (1-PearsonCorrelation))
data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void UpdateProfile(const data_t PearsonCorrelation, const index_t row, const index_t column,
			       stream<data_t, 5> &rowWiseAggregate_in, stream<data_t, 5> &columnWiseAggregate_in,
                   stream<index_t, 5> &rowWiseIndex_in, stream<index_t, 5> &columnWiseIndex_in,
                   stream<data_t, 5> &rowWiseAggregate_out, stream<data_t, 5> &columnWiseAggregate_out,
                   stream<index_t, 5> &rowWiseIndex_out, stream<index_t, 5> &columnWiseIndex_out) {
    data_t rowAggregate = rowWiseAggregate_in.read();
    index_t rowIndex = rowWiseIndex_in.read();

    data_t columnAggregate = columnWiseAggregate_in.read();
    index_t columnIndex = columnWiseIndex_in.read();

    // (i-m/4 ≤ j ≤ i+m/4)
    bool exclusionZoneRow = (row - static_cast<int>(m) / 4) <= column && column <= (row + static_cast<int>(m) / 4);

    if (!exclusionZoneRow && PearsonCorrelation > rowAggregate) {
        // update to "remember" best column in the current row
        rowWiseAggregate_out.write(PearsonCorrelation);
        rowWiseIndex_out.write(column);
    } else {
        // push (previous) row aggregate forward (i.e. do not update value)
        rowWiseAggregate_out.write(rowAggregate);
        rowWiseIndex_out.write(rowIndex);
    }

    // (j-m/4 ≤ i ≤ j+m/4)
    bool exclusionZoneColumn = (column - static_cast<int>(m) / 4) <= row && row <= (column + static_cast<int>(m) / 4);

    if (!exclusionZoneColumn && PearsonCorrelation > columnAggregate) {
        // update to "remember" best row in the current column
        columnWiseAggregate_out.write(PearsonCorrelation);
        columnWiseIndex_out.write(row);
    } else {
        // push (previous) column aggregate forward (i.e. do not update value)
        columnWiseAggregate_out.write(columnAggregate);
        columnWiseIndex_out.write(columnIndex);
    }

}

void ComputeMatrixProfile(const size_t stage,
                          stream<data_t, 5> &QT_in, stream<data_t, 5> &df_i_in, stream<data_t, 5> &dg_i_in,
                          stream<data_t, 5> &df_j_in, stream<data_t, 5> &dg_j_in, stream<data_t, 5> &inv_i_in, stream<data_t, 5> &inv_j_in,

			              stream<data_t, 5> &rowWiseAggregate_in, stream<data_t, 5> &columnWiseAggregate_in,
                          stream<index_t, 5> &rowWiseIndex_in, stream<index_t, 5> &columnWiseIndex_in,

			              stream<data_t, 5> &QT_out, stream<data_t, 5> &df_i_out, stream<data_t, 5> &dg_i_out,
                          stream<data_t, 5> &df_j_out, stream<data_t, 5> &dg_j_out, stream<data_t, 5> &inv_i_out, stream<data_t, 5> &inv_j_out,

			              stream<data_t, 5> &rowWiseAggregate_out, stream<data_t, 5> &columnWiseAggregate_out,
			              stream<index_t, 5> &rowWiseIndex_out, stream<index_t, 5> &columnWiseIndex_out) {

    // Push Current Aggregate along (not affected by current compute unit)
    for (int i = 0; i < stage; ++i) {
        data_t aggregate = columnWiseAggregate_in.read();
        columnWiseAggregate_out.write(aggregate);
        index_t aggregateIndex = columnWiseIndex_in.read();
        columnWiseIndex_out.write(aggregateIndex);
    }

    // The initial value of QT for the current compute unit, i.e. Q_{i-1, j-1} in formulas below
    data_t QT = QT_in.read();

    data_t inv_i = inv_i_in.read();
    data_t inv_j = inv_j_in.read();

    // calculate pearson correlation
    data_t PearsonCorrelation = QT * inv_i * inv_j;

    index_t row = 0, column = stage;
    UpdateProfile(PearsonCorrelation, row, column, rowWiseAggregate_in, columnWiseAggregate_in,
                  rowWiseIndex_in, columnWiseIndex_in, rowWiseAggregate_out, columnWiseAggregate_out, rowWiseIndex_out,
                  columnWiseIndex_out);

    // Push along the pipeline
    inv_i_out.write(inv_i);

    data_t QT_fwd;

    for (int t = 0; t < (n - m + 1) - 1 - stage; ++t) {
        // Forward QT to the next stage
        QT_fwd = QT_in.read();
        QT_out.write(QT_fwd);

        // Calculate QT Values for the current compute unit

        // Corresponds to df_i
        data_t df_i = df_i_in.read();

        // Corresponds to df_j
        data_t df_j = df_j_in.read();

        // Corresponds to dg_i
        data_t dg_i = dg_i_in.read();

        // Corresponds to dg_j
        data_t dg_j = dg_j_in.read();

        // Calculate new values of QT: QT_{i, j} = QT_{i-1, j-1} - df_{i} * dg_{j} + df_{j} * dg_{i}
        QT = QT + df_i * dg_j + df_j * dg_i;

        // Use QT to calculate Pearson Correlation

        // Corresponds to 1 / (||T_{i, m} - μ[i]||)
        inv_i = inv_i_in.read();

        // Corresponds to 1 / (||T_{j, m} - μ[j]||)
        inv_j = inv_j_in.read();

        // Calculate Pearson Correlation: QT * (1 / (||T_{i, m} - μ[i]||)) * (1 /
        // (||T_{j, m} - μ[j]||))
        PearsonCorrelation = QT * inv_i * inv_j;

        // Pass on aggregates (update if required)
        row = t + 1; column = stage + t + 1;
        UpdateProfile(PearsonCorrelation, row, column, rowWiseAggregate_in, columnWiseAggregate_in,
                      rowWiseIndex_in, columnWiseIndex_in, rowWiseAggregate_out, columnWiseAggregate_out,
                      rowWiseIndex_out, columnWiseIndex_out);

        // can push everything because first element not in the loop
        inv_j_out.write(inv_j);

        // Move elements along the pipeline
        if (t != (n - m + 1) - 1 - stage - 1) {
            // Push Everything Except last i
            df_i_out.write(df_i);
            dg_i_out.write(dg_i);
            inv_i_out.write(inv_i);
        }

        if (t != 0) {
            // Push Everything Except first j
            df_j_out.write(df_j);
            dg_j_out.write(dg_j);
        }

    }

    // Push Aggregate along
    for (int i = 0; i < stage; ++i) {
        data_t aggregate = rowWiseAggregate_in.read();
        rowWiseAggregate_out.write(aggregate);
        index_t aggregateIndex = rowWiseIndex_in.read();
        rowWiseIndex_out.write(aggregateIndex);
    }

}

void StreamToMemory(stream<data_t, 5> &rowWiseAggregate, stream<data_t, 5> &columnWiseAggregate,
                    stream<index_t, 5> &rowWiseIndex, stream<index_t, 5> &columnWiseIndex,
			        data_t *MP, index_t *MPI) {
    for (int i = 0; i < rs_len; ++i) {
        data_t rowAggregate = rowWiseAggregate.read();
        index_t row = rowWiseIndex.read();

        data_t columnAggregate = columnWiseAggregate.read();
        index_t column = columnWiseIndex.read();

        // Take the max of both row/column & Convert from PearsonCorrelation to Euclidean Distance
        if (rowAggregate > columnAggregate) {
            MP[i] = PearsonCorrelationToEuclideanDistance(rowAggregate);
            MPI[i] = row;
        } else {
            MP[i] = PearsonCorrelationToEuclideanDistance(columnAggregate);
            MPI[i] = column;
        }
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    #pragma HLS DATAFLOW
    const size_t numStages = rs_len;

    // Streams required to calculate Correlations
    stream<data_t, 5> QT[numStages + 1];

    stream<data_t, 5> df_i[numStages + 1];
    stream<data_t, 5> df_j[numStages + 1];

    stream<data_t, 5> dg_i[numStages + 1];
    stream<data_t, 5> dg_j[numStages + 1];

    stream<data_t, 5> inv_i[numStages + 1];
    stream<data_t, 5> inv_j[numStages + 1];

    // Store the intermediate results
    stream<data_t, 5> rowWiseAggregate[numStages + 1];
    stream<data_t, 5> columnWiseAggregate[numStages + 1];

    stream<index_t, 5> rowWiseIndex[numStages + 1];
    stream<index_t, 5> columnWiseIndex[numStages + 1];

    MemoryToStream(T, QT[0], df_i[0], df_j[0], dg_i[0], dg_j[0], inv_i[0], inv_j[0],
                   rowWiseAggregate[0], columnWiseAggregate[0], rowWiseIndex[0], columnWiseIndex[0]);

    for (int k = 0; k < rs_len; ++k){
        #pragma HLS UNROLL
        ComputeMatrixProfile(k, QT[k], df_i[k], dg_i[k], df_j[k], dg_j[k], inv_i[k], inv_j[k],
                             rowWiseAggregate[k], columnWiseAggregate[k], rowWiseIndex[k], columnWiseIndex[k],
                             QT[k + 1], df_i[k + 1], dg_i[k + 1], df_j[k + 1], dg_j[k + 1], inv_i[k + 1], inv_j[k + 1],
                             rowWiseAggregate[k + 1], columnWiseAggregate[k + 1], rowWiseIndex[k + 1], columnWiseIndex[k + 1]);
    }

    StreamToMemory(rowWiseAggregate[numStages], columnWiseAggregate[numStages],
                   rowWiseIndex[numStages], columnWiseIndex[numStages], MP, MPI);

    // Deplete Pipeline Completely (inv_i is always passed along, needs to be "popped" explicitly)
    inv_i[numStages].read();
}
