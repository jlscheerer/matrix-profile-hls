/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#include "Config.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_math.h"

void PrecomputationProcessingElement(const data_t *T, data_t (&mu)[sublen], data_t (&df)[sublen], data_t (&dg)[sublen], data_t (&inv)[sublen], 
                                     data_t (&QT)[sublen], data_t (&P)[sublen], aggregate_t (&rowAggregate)[sublen], aggregate_t (&columnAggregate)[sublen]) {
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
    data_t mu0 = mean; mu[0] = mu0; df[0] = 0; dg[0] = 0;

    data_t inv_sum = 0, qt_sum = 0;
    PrecomputationInitInvQT:
    for (size_t k = 0; k < m; ++k) {
        #pragma HLS UNROLL
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }

    data_t inv0 = 1 / sqrt(inv_sum);
    inv[0] = inv0; QT[0] = qt_sum; P[0] = 1;

    // Assumption: will always be in the exclusionZone
    columnAggregate[0] = aggregate_t_init;

    // Maximum PearsonCorrelation and corresponding Index for the first row
    aggregate_t rowAggregate_m = aggregate_t_init;

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
        data_t prev_mean = mean;
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

        rowAggregate[i - m + 1] = aggregate_t_init;
        
        bool exclusionZone = (i - m + 1) <= m / 4;
        if(!exclusionZone) columnAggregate[i - m + 1] = aggregate_t_init;
        else columnAggregate[i - m + 1] = {P[i - m + 1], 0};

        if (!exclusionZone && P[i - m + 1] > rowAggregate_m.value)
            rowAggregate_m = {P[i - m + 1], static_cast<index_t>(i - m + 1)};

        // shift all values in T_m back
        PrecomputationComputeShift: 
        for (size_t k = 0; k < m - 1; ++k){
            #pragma HLS UNROLL
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = T_i;
    }
    // set the aggregates for the first row
    rowAggregate[0] = rowAggregate_m;
}

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    #pragma HLS INLINE
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void ReductionComputionElement(aggregate_t (&rowAggregate)[sublen], aggregate_t (&columnAggregate)[sublen], data_t *MP, index_t *MPI) {
    #pragma HLS INLINE
    // Just always take the max
    ReductionCompute:
    for (size_t i = 0; i < sublen; ++i) {
        #pragma HLS PIPELINE II=1
        aggregate_t rowAggregate_m = rowAggregate[i], columnAggregate_m = columnAggregate[i];
        // Take the max and compute EuclideanDistance
        MP[i]  = PearsonCorrelationToEuclideanDistance(rowAggregate_m.value > columnAggregate_m.value ? rowAggregate_m.value : columnAggregate_m.value);
        MPI[i] = rowAggregate_m.value > columnAggregate_m.value ? rowAggregate_m.index : columnAggregate_m.index;
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

    aggregate_t rowAggregate[sublen], columnAggregate[sublen];
    // factor=3 required for fadd and fmul (update if data_t changes)
    #pragma HLS ARRAY_PARTITION variable=columnAggregate      cyclic factor=3

    PrecomputationProcessingElement(T, mu, df, dg, inv, QT, P, rowAggregate, columnAggregate);

    // Do the actual calculations via updates
    MatrixProfileComputeRow:
    for (size_t row = 1; row < sublen; ++row) {
        data_t dfi = df[row]; data_t dgi = dg[row]; data_t invi = inv[row];
        aggregate_t rowAggregate_m = aggregate_t_init;

        // exclusionZone integrated into loop bounds
        // exclusionZone <==> row - m/4 <= column <= row + m/4
        //               <==> column <= row + m/4 [(row <= column, m > 0) ==> row - -m/4 <= column]
        //               <==> row + k <= row + m/4
        //               <==> k <= m/4
        MatrixProfileComputeColumn:
        for (size_t k = (m / 4) + 1; k < sublen - row; ++k) {
            #pragma HLS PIPELINE II=1
            // QT_{i, j} = QT_{i-1, j-1} + df_i * dg_j + df_j * dg_i
            // QT[k] was the previous value (i.e. value diagonally above the current QT[k])
            QT[k] = QT[k] + dfi * dg[k + row] + df[k + row] * dgi;
            // calculate pearson correlation
            // P_{i, j} = QT_{i, j} * inv_i * inv_j
            P[k] = QT[k] * invi * inv[k + row];

            // Update Aggregates
            const size_t column = row + k;
            if(P[k] > columnAggregate[column].value)
                columnAggregate[column] = {P[k], static_cast<index_t>(row)};
            if(P[k] > rowAggregate_m.value)
                rowAggregate_m = {P[k], static_cast<index_t>(column)};
        }
        rowAggregate[row] = rowAggregate_m;
    }
    
    ReductionComputionElement(rowAggregate, columnAggregate, MP, MPI);
}
