/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_math.h"

void PrecomputationProcessingElement(const data_t *T, data_t (&mu)[sublen], data_t (&df)[sublen], data_t (&dg)[sublen], data_t (&inv)[sublen], 
                                     data_t (&QT)[sublen], data_t (&P)[sublen], data_t (&rowAggregate)[sublen], index_t (&rowAggregateIndex)[sublen],
                                     data_t (&columnAggregate)[sublen], index_t (&columnAggregateIndex)[sublen]) {
    #pragma HLS INLINE
    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];
    #pragma HLS ARRAY_PARTITION variable=T_m complete

    // the first m T values, required for convolution
    data_t Ti_m[m];
    #pragma HLS ARRAY_PARTITION variable=Ti_m complete

    PrecomputationInitT:
    for (size_t i = 0; i < m; ++i) {
        #pragma HLS PIPELINE II=1
        data_t T_i = T[i];
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }

    data_t mean = 0;
    PrecomputationInitMu:
    for(size_t i = 0; i < m; ++i){
        #pragma HLS UNROLL
        mean += T_m[i];
    }
    mean /= m;

    // calculate initial values
    data_t mu0 = mean;
    mu[0] = mu0;
    df[0] = 0;
    dg[0] = 0;

    data_t inv_sum = 0;
    data_t qt_sum = 0;
    PrecomputationInitInvQT:
    for (size_t k = 0; k < m; ++k) {
        #pragma HLS UNROLL
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }

    data_t inv0 = 1 / sqrt(inv_sum);
    inv[0] = inv0;
    QT[0] = qt_sum;
    P[0] = 1;

    rowAggregate[0] = aggregate_init; rowAggregateIndex[0] = index_init;
    columnAggregate[0] = aggregate_init; columnAggregateIndex[0] = index_init;

    data_t prev_mean;
    PrecomputationCompute:
    for (size_t i = m; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        data_t T_i = T[i];
        data_t T_r = T_m[0];

        // recompute mean to achieve II=1
        mean = 0;
        PrecomputationComputeUpdateMean:
        for(size_t k = 1; k < m; ++k) {
            #pragma HLS UNROLL
            mean += T_m[k];
        }
        prev_mean = mean;
        prev_mean += T_r; prev_mean /= m;
        mean += T_i; mean /= m;

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        df[i - m + 1] = (T_i - T_r) / 2;
        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        dg[i - m + 1] = (T_i - mean) + (T_r - prev_mean);

        inv_sum = 0;
        qt_sum = 0;

        PrecomputationComputeUpdateInvQT:
        for (size_t k = 1; k < m; k++) {
            #pragma HLS UNROLL
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

        rowAggregate[i - m + 1] = aggregate_init; rowAggregateIndex[i - m + 1] = index_init;
        columnAggregate[i - m + 1] = aggregate_init; columnAggregateIndex[i - m + 1] = index_init;

        // shift all values in T_m back
        PrecomputationComputeShift: 
        for (size_t k = 0; k < m - 1; ++k){
            #pragma HLS UNROLL
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = T_i;
    }
}

void UpdateAggregates(size_t row, data_t (&P)[sublen], data_t (&rowAggregate)[sublen], index_t (&rowAggregateIndex)[sublen],
                      data_t (&columnAggregate)[sublen], index_t (&columnAggregateIndex)[sublen]) {
    // P each iteration (row) P contains one less valid value (upper-triangular matrix)
    data_t rowMax = aggregate_init; index_t rowMaxIndex = index_init;

    UpdateAggregateCompute:
    for (size_t column = row; column < sublen; ++column) {
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
    #pragma HLS INLINE
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void ReductionComputionElement(data_t (&rowAggregate)[sublen], index_t (&rowAggregateIndex)[sublen],
                               data_t (&columnAggregate)[sublen], index_t (&columnAggregateIndex)[sublen], data_t *MP, index_t *MPI) {
    #pragma HLS INLINE
    // Just always take the max
    ReductionCompute:
    for (size_t i = 0; i < sublen; ++i) {
        data_t rowValue = rowAggregate[i]; index_t rowIndex = rowAggregateIndex[i];
        data_t colValue = columnAggregate[i]; index_t colIndex = columnAggregateIndex[i];
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
    #pragma HLS INTERFACE m_axi     port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi     port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=MPI offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=T   bundle=control
    #pragma HLS INTERFACE s_axilite port=MP  bundle=control
    #pragma HLS INTERFACE s_axilite port=MPI bundle=control

    data_t mu[sublen], df[sublen], dg[sublen], inv[sublen];
    data_t QT[sublen], P[sublen];

    data_t rowAggregate[sublen]; index_t rowAggregateIndex[sublen];
    data_t columnAggregate[sublen]; index_t columnAggregateIndex[sublen];

    PrecomputationProcessingElement(T, mu, df, dg, inv, QT, P, rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex);

    // TODO: Could move this inside the Precomputation
    // update/initialize Aggregates for the first row
    UpdateAggregates(0, P, rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex);

    // Do the actual calculations via updates
    MatrixProfileComputeRow:
    for (size_t row = 1; row < sublen; ++row) {
        data_t dfi = df[row]; data_t dgi = dg[row]; data_t invi = inv[row];
        
        MatrixProfileComputeColumn:
        for (size_t k = 0; k < sublen - row; ++k) {
            // QT_{i, j} = QT_{i-1, j-1} + df_i * dg_j + df_j * dg_i
            // QT[k] was the previous value (i.e. value diagonally above the current QT[k])
            QT[k] = QT[k] + dfi * dg[k + row] + df[k + row] * dgi;
            // calculate pearson correlation
            // P_{i, j} = QT_{i, j} * inv_i * inv_j
            P[k] = QT[k] * invi * inv[k + row];
        }

        // Update Aggregates for the current row
        // TODO: Instead of this inline function directly into upper body
        UpdateAggregates(row, P, rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex);
    }
    
    ReductionComputionElement(rowAggregate, rowAggregateIndex, columnAggregate, columnAggregateIndex, MP, MPI);
}