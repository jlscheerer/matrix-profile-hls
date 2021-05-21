/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_math.h"

void PrecomputationProcessingElement(const data_t *T, data_t (&mu)[rs_len], data_t (&df)[rs_len], data_t (&dg)[rs_len], data_t (&inv)[rs_len], 
                                     data_t (&QT)[rs_len], data_t (&P)[rs_len], data_t (&rowAggregate)[rs_len], index_t (&rowAggregateIndex)[rs_len],
                                     data_t (&columnAggregate)[rs_len], index_t (&columnAggregateIndex)[rs_len]) {
    data_t mean = 0;
    data_t inv_sum = 0;
    data_t qt_sum = 0;

    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];

    // the first m T values, required for convolution
    data_t Ti_m[m];

    PrecomputationInitTM:
    for (size_t i = 0; i < m; ++i) {
        data_t T_i = T[i];
        mean += T_i;
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }
    mean /= m;

    // calculate initial values
    data_t mu0 = mean;
    mu[0] = mu0;
    df[0] = 0;
    dg[0] = 0;

    // TODO: unroll this loop
    PrecomputationInitInvQT:
    for (size_t k = 0; k < m; ++k) {
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }

    data_t inv0 = 1 / sqrt(inv_sum);
    inv[0] = inv0;
    QT[0] = qt_sum;
    P[0] = 1;

    rowAggregate[0] = aggregate_init;
    rowAggregateIndex[0] = index_init;
    columnAggregate[0] = aggregate_init;
    columnAggregateIndex[0] = index_init;

    data_t prev_mean;
    
    PrecomputationCompute:
     for (size_t i = m; i < n; ++i) {
        data_t T_i = T[i];
        data_t T_r = T_m[0];

        // set and update mean
        prev_mean = mean;
        mean = mean + (T_i - T_r) / m;
        mu[i - m + 1] = mean;

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        df[i - m + 1] = (T_i - T_r) / 2;
        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        dg[i - m + 1] = (T_i - mean) + (T_r - prev_mean);

        inv_sum = 0;
        qt_sum = 0;

        // needs to be unrolled but then not to difficult
        PrecomputationComputeUpdateInvQT:
        for (size_t k = 1; k < m; k++) {
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }

        // perform last element of the loop separately (this requires the new value)
        inv_sum += (T_i - mean) * (T_i - mean);
        inv[i - m + 1] = 1 / sqrt(inv_sum);

        qt_sum += (T_i - mean) * (Ti_m[m - 1] - mu0);
        QT[i - m + 1] = qt_sum;

        // calculate Pearson Correlation: P_{i, j} = QT_{i, j} * inv_i * inv_j
        P[i - m + 1] = qt_sum * inv0 * 1 / sqrt(inv_sum);

        rowAggregate[i - m + 1] = aggregate_init;
        rowAggregateIndex[i - m + 1] = index_init;
        columnAggregate[i - m + 1] = aggregate_init;
        columnAggregateIndex[i - m + 1] = index_init;

        // shift all values in T to the left (if unrolled not difficult)
        PrecomputationComputeShift: 
        for (size_t k = 0; k < m - 1; ++k){
            T_m[k] = T_m[k + 1];
        }

        T_m[m - 1] = T_i;
    }
}

void UpdateAggregates(size_t row, data_t (&P)[rs_len], data_t (&rowAggregate)[rs_len], index_t (&rowAggregateIndex)[rs_len],
                      data_t (&columnAggregate)[rs_len], index_t (&columnAggregateIndex)[rs_len]) {
    // P each iteration (row) P contains one less valid value (upper-triangular matrix)
    data_t rowMax = aggregate_init;
    index_t rowMaxIndex = index_init;

    UpdateAggregateCompute:
    for (size_t column = row; column < n - m + 1; ++column) {
        // check if we are in the exclusion Zone
        // exlusionZone <==> row - m/4 <= column <= row + m/4
        // 				<==> column <= row + m/4 [(row <= column, m > 0) ==> row - -m/4 <= column]
        bool exlusionZone = column <= row + m / 4;
        if (!exlusionZone && P[column - row] > columnAggregate[column]) {
            columnAggregate[column] = P[column - row];
            columnAggregateIndex[column] = row;
        }
        if (!exlusionZone && P[column - row] > rowMax) {
            rowMax = P[column - row];
            rowMaxIndex = column;
        }
    }

    rowAggregate[row] = rowMax;
    rowAggregateIndex[row] = rowMaxIndex;
}

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void ReductionComputionElement(data_t (&rowAggregate)[rs_len], index_t (&rowAggregateIndex)[rs_len],
                               data_t (&columnAggregate)[rs_len], index_t (&columnAggregateIndex)[rs_len], data_t *MP, index_t *MPI) {
    // Just always take the max
    ReductionCompute:
    for (size_t i = 0; i < n - m + 1; ++i) {
        data_t rowValue = rowAggregate[i];
        index_t rowIndex = rowAggregateIndex[i];
        data_t colValue = columnAggregate[i];
        index_t colIndex = columnAggregateIndex[i];
        // Take the max and compute EuclideanDistance
        if (rowValue > colValue) {
            MP[i] = PearsonCorrelationToEuclideanDistance(rowValue);
            MPI[i] = rowIndex;
        } else {
            MP[i] = PearsonCorrelationToEuclideanDistance(colValue);
            MPI[i] = colIndex;
        }
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    data_t mu[rs_len], df[rs_len], dg[rs_len], inv[rs_len];
    data_t QT[rs_len], P[rs_len];

    data_t rowAggregate[n - m + 1], columnAggregate[n - m + 1];
    index_t rowAggregateIndex[n - m + 1], columnAggregateIndex[n - m + 1];

    PrecomputationProcessingElement(T, mu, df, dg, inv, QT, P, rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex);

    // TODO: Could move this inside the Precomputation
    // Update/Initialize Aggregates for the first row
    UpdateAggregates(0, P, rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex);

    // Do the actual calculations via updates
    MatrixProfileComputeRow:
    for (size_t row = 1; row < n - m + 1; ++row) {
        
        data_t dfi = df[row]; data_t dgi = dg[row]; data_t invi = inv[row];
        
        MatrixProfileComputeColumn:
        for (size_t k = 0; k < n - m + 1 - row; ++k) {
            // column = k + row
            // QT_{i, j} = QT_{i-1, j-1} + df_i * dg_j + df_j * dg_i
            // QT[k] was the previous value (i.e. value diagonally above the current QT[k])
            QT[k] = QT[k] + dfi * dg[k + row] + df[k + row] * dgi;
            // Directly already calculate the pearson correlation
            // P_{i, j} = QT_{i, j} * inv_i * inv_j
            P[k] = QT[k] * invi * inv[k + row];
        }

        // Update Aggregates for the current row
        // TODO: Instead of this inline function directly into upper body
        UpdateAggregates(row, P, rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex);
    }
    
    ReductionComputionElement(rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex, MP, MPI);
}