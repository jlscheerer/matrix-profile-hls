/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    
    #include "kernel/MatrixProfileKernel.hpp"
    
    #include "hls_math.h"
#endif

#include "kernel/DataPacks.hpp"
#include "kernel/TreeReduce.hpp"

void MatrixProfileKernelTLF(const InputDataPack *in, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem1

    data_t QT[n - m + 1];
    DataPack columnData[n - m + 1], rowData[n - m + 1];
    
    // Store computed row aggregates to merge with column 
    // aggregates during the final reduction stage
    aggregate_t rowAggregates[n - m + 1];

    MatrixProfileInit:
    for (index_t i = 0; i < n - m + 1; ++i) {
       	#pragma HLS PIPELINE II=1
        const InputDataPack read = in[i];
        const DataPack data = DataPack{read.df, read.dg, read.inv};

        // Store read QT value in seperate arary will be updated
        // during the actual computation
        QT[i] = read.QT;

        // explicitely store two copies of the input data to
        // later access both independently in a single cycle 
        rowData[i] = data;
        columnData[i] = data;
    }

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
    constexpr int T = 2;
    aggregate_t columnReduce[n - m + 1][T];

    // =============== [/Compute] ===============
    MatrixProfileComputeRow:
    for (index_t k = 0; k < n - m + 1; ++k) {
        MatrixProfileComputeColumn:
        for (index_t i = 0; i < n - m + 1; ++i) {
            #pragma HLS PIPELINE II=1
            const index_t columnIndex = k + i;
            const bool computationInRange = k + i < n - m + 1;
            const bool exclusionZone = i < (m / 4);

            const DataPack row = rowData[k];
            const DataPack column = (!exclusionZone && computationInRange)
                                            ? columnData[columnIndex] : (DataPack){0, 0, 0};

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
            aggregate_t prevColumn = (k < T/2) ? aggregate_t_init : columnReduce[columnIndex % (n - m + 1)][k % T];
	        columnReduce[columnIndex % (n - m + 1)][(k + T/2) % T] = prevColumn.value > P ? prevColumn : aggregate_t(P, k);
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
	    const aggregate_t columnAggregate = TreeReduce::Maximum<aggregate_t, T>(columnReduce[i]);
        // Compute Maximum of both aggregates (i.e. values with lowest EuclideanDistance)
        const aggregate_t aggregate = rowAggregate.value > columnAggregate.value
					? rowAggregate : columnAggregate;
        // Store the Result
        MP[i] = aggregate.value;
        MPI[i] = aggregate.index;
    }
    // =============== [/Reduce] ===============
}
