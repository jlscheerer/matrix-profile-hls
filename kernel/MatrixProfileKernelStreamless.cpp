/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    
    #include "kernel/MatrixProfileKernel.hpp"
    #include "kernel/TreeReduce.hpp"

    #include "hls_math.h"
#endif


data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    #pragma HLS INLINE
    return sqrt(static_cast<data_t>(2 * m * (1 - PearsonCorrelation)));
}

// Structure containing all values required for Update
// Computation and Conversion to PearsonCorrelation
struct ComputePack { data_t df, dg, inv; };

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem2

    data_t QT[sublen];
    ComputePack columnData[n - m + 1], rowData[n - m + 1];

    // Store computed row aggregates to merge with column 
    // aggregates during the final reduction stage
    aggregate_t rowAggregates[n - m + 1];

    // =============== [Precompute] ===============
    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];
    #pragma HLS ARRAY_PARTITION variable=T_m complete

    // the first m T values, required for convolution
    data_t Ti_m[m];
    #pragma HLS ARRAY_PARTITION variable=Ti_m complete

    data_t mu0 = 0, inv_sum = 0, qt_sum = 0;
    PrecomputationInitTMu:
    for (index_t i = 0; i < m; ++i) {
        #pragma HLS PIPELINE II=1
	data_t T_i = T[i];
        mu0 += T_i;
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }
    mu0 /= m;

    PrecomputationInitInvQT:
    for (index_t k = 0; k < m; ++k) {
	#pragma HLS PIPELINE II=1
        inv_sum += (T_m[k] - mu0) * (T_m[k] - mu0);
        qt_sum += (T_m[k] - mu0) * (Ti_m[k] - mu0);
    }

    data_t inv0 = static_cast<data_t>(1) / sqrt(inv_sum);

    const ComputePack compute0 = {0, 0, inv0};
    columnData[0] = compute0; rowData[0] = compute0;
    QT[0] = qt_sum;

    PrecomputationCompute:
    for (index_t i = m; i < n; ++i) {
	#pragma HLS PIPELINE II=1
        data_t T_i = T[i];
        data_t T_r = T_m[0];

	// recompute means to break dependency
	// and therefore achieve lower II
        const data_t prev_mean = TreeReduce::Add<data_t, m>(T_m) / m;
        const data_t mean = prev_mean + (T_i - T_r) / m;

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        const data_t df = (T_i - T_r) / 2;

        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        const data_t dg = (T_i - mean) + (T_r - prev_mean);

        inv_sum = 0; qt_sum = 0;
        PrecomputationComputeUpdateInvQT:
        for (index_t k = 1; k < m; k++) {
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }

        // perform last element of the loop separately (this requires the new value)
        inv_sum += (T_i - mean) * (T_i - mean);
        qt_sum += (T_i - mean) * (Ti_m[m - 1] - mu0);

        const data_t inv = static_cast<data_t>(1) / sqrt(inv_sum);

        const ComputePack compute = {df, dg, inv};
        QT[i - m + 1] = qt_sum;
        columnData[i - m + 1] = compute; rowData[i - m + 1] = compute;

        // shift all values in T_m back
        PrecomputationComputeShift: 
        for (index_t k = 0; k < m - 1; ++k)
            T_m[k] = T_m[k + 1];
        T_m[m - 1] = T_i;
    }

    // =============== [/Precompute] ===============

    // TODO: Comment Breaking Dependency by Introducing "Delay-Buffer"
    aggregate_t rowReduce[16];
    #pragma HLS ARRAY_PARTITION variable=rowReduce complete

    // rowReduce needs to be explicitly initialized if n - m + 1 < 16
    // Because for every row we perform a reduction on all elements!
    // For every n - m + 1 >= 16 we handle this via implicit initializaiton
    for (index_t i = 0; i < 16; ++i) {
	#pragma HLS UNROLL
	rowReduce[i] = aggregate_t_init;
    }

    // TODO: Comment "Double-Buffer" Write to opposite position than is being read
    // TODO: Clean-Up Access to columnReduce
    constexpr int TBuf = 2;
    aggregate_t columnReduce[n - m + 1][TBuf];

    // =============== [/Compute] ===============
    MatrixProfileComputeRow:
    for (index_t k = 0; k < n - m + 1; ++k) {
        MatrixProfileComputeColumn:
        for (index_t i = 0; i < n - m + 1; ++i) {
            #pragma HLS PIPELINE II=1
            const index_t columnIndex = k + i;
            const bool computationInRange = k + i < n - m + 1;
            const bool exclusionZone = i < (m / 4);

            const ComputePack row = rowData[k];
            const ComputePack column = (!exclusionZone && computationInRange)
                                            ? columnData[columnIndex] : (ComputePack){0, 0, 0};

	        // Update QT value diagonally above via the update formulation
            // QT_{i, j} = QT_{i-1, j-1} + df_i * dg_j + df_j * dg_i
            QT[i] += row.df * column.dg + column.df * row.dg;

            // Calculate PearsonCorrelation
            // P_{i, j} = QT_{i, j} * inv_i * inv_j
            const data_t P = QT[i] * row.inv * column.inv;

            // Row-Wise Partial Reduction
            aggregate_t prevRow = (i < 16) ? aggregate_t_init : rowReduce[i % 16];
            rowReduce[i % 16] = P > prevRow.value ? aggregate_t(P, columnIndex) : prevRow;

            // Column-Wise Partial Reduction
            // Wrap-Around works because if we are not outside 
            // the exlusionZone P will be 0 and therefore irrelevant
            aggregate_t prevColumn = (k < TBuf/2) ? aggregate_t_init : columnReduce[columnIndex % (n - m + 1)][k % TBuf];
	    columnReduce[columnIndex % (n - m + 1)][(k + TBuf/2) % TBuf] = prevColumn.value > P ? prevColumn : aggregate_t(P, k);
        }

        // Only need to reduce a constant number (16) of 
        // aggregates via guaranteed TreeReduction
        rowAggregates[k] = TreeReduce::Maximum<aggregate_t, 16>(rowReduce);
    }
    // =============== [/Compute] ===============
    
    // =============== [Reduce] ===============

    // Compute maximum between Row- and Column-wise aggregates
    ReductionCompute:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
	    // Load rowAggregate we computed directly
        const aggregate_t rowAggregate = rowAggregates[i];
        // Compute columnAggregate by performing TreeReduction on partial results
	    const aggregate_t columnAggregate = TreeReduce::Maximum<aggregate_t, TBuf>(columnReduce[i]);
        // Compute Maximum of both aggregates (i.e. values with lowest EuclideanDistance)
        const aggregate_t aggregate = rowAggregate.value > columnAggregate.value
					? rowAggregate : columnAggregate;
        // Store the Result
        MP[i] = PearsonCorrelationToEuclideanDistance(aggregate.value);
        MPI[i] = aggregate.index;
    }
    // =============== [/Reduce] ===============
}
