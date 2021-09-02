/**
 * @file    MatrixProfileKernelVanilla.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Vanilla-Kernel]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"
    #include "hls_math.h"
#endif

#include "kernel/DataPacks.hpp"
#include "kernel/TreeReduce.hpp"

void MatrixProfileKernelTLF(const index_t n, const index_t m, const index_t iteration,
                            const InputDataPack *input, OutputDataPack *output) {
    #pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

    // Determine values corresponding to the current iteration
    // In particular, the number of rows of the current DiagonalChunk
    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    // QT values of the current DiagonalChunk
    data_t QT[nColumns];

    constexpr int rowReduceD1 = 8;
    constexpr int rowReduceD2 = 8;

    // Buffer used for partial reduction of rowAggregates
    // As opposed to the Thesis we use a two-dimensional buffer
    // to lower memory requirements of the TreeReduction
    aggregate_t rowReduce[rowReduceD1][rowReduceD2];
    #pragma HLS ARRAY_PARTITION variable=rowReduce dim=2 complete

    // Instead of shifting the column values explicitly like in the Tiled Kernel
    // We use a "double-buffering" technique
    aggregate_t columnAggregates[nColumns][2];

    // Initialize our internal QT buffer using data provided by the Driver
    MatrixProfileInitQT:
    for (index_t i = 0; i < nColumns; ++i) {
	#pragma HLS PIPELINE II=1

        // Determine the current column index
        const index_t columnIndex = nOffset + i;
        // Check whether the index is still contained in the input data
        const bool inBounds = columnIndex < n - m + 1;
        // If we are in bounds, take the values, otherwise use 0
        const InputDataPack read = inBounds ? input[columnIndex]
                                            : InputDataPack(0);
        QT[i] = read.QT;
    }
	
    MatrixProfileComputeRow:
    for (index_t i = 0; i < nRows; ++i) {
        // Determine the current row index, as well as the associated data
        const index_t rowIndex = i;
        const InputDataPack row = input[rowIndex];

        MatrixProfileComputeTile:
        for (index_t j = 0; j < nColumns; ++j) {
	        #pragma HLS PIPELINE II=1

            // Determine the corresponding column index
            const index_t columnIndex = nOffset + i + j;

            // Check whether the calculated values is relevant for the matrix 
            // profile, i.e., is both in bounds and not in the exclusion zone
            const bool columnInBounds = columnIndex < n - m + 1;
            const bool exclusionZone = rowIndex > columnIndex - m / 4;
            const bool inBounds = columnInBounds && !exclusionZone;

            // Get the column data for the j-th diagonal
            const InputDataPack column = columnInBounds ? input[columnIndex]
                                                        : InputDataPack(0);

            // Apply the SCAMP update formulation
            QT[j] += row.df * column.dg + column.df * row.dg;

            // In case the computed value is not relevant take 0 (will not affect the result)
            const data_t P = inBounds ? static_cast<data_t>(QT[j] * row.inv * column.inv) : static_cast<data_t>(0);

            // Use partial reduction of rowAggregates to break loop dependency
            const aggregate_t prevRow = (j < rowReduceD2) ? aggregate_t_init 
                                                          : rowReduce[i % rowReduceD1][j % rowReduceD2];
            rowReduce[i % rowReduceD1][j % rowReduceD2] = prevRow.value > P ? prevRow 
                                                                            : aggregate_t(P, columnIndex);

            // Implicit initialization of columnAggregates, simultaneous update and implicit shift
            const aggregate_t prevColumn = (j < nColumns - 1 && i > 0) ? columnAggregates[j + 1][i % 2] 
                                                                       : aggregate_t_init;
            columnAggregates[j][(i + 1) % 2] = P > prevColumn.value ? aggregate_t(P, rowIndex) 
                                                                    : prevColumn;
        }
        
        // determine the row-wise aggregate using TreeReduction
        const aggregate_t rowAggregate = TreeReduce::Maximum<aggregate_t, rowReduceD2>(rowReduce[i % rowReduceD1]);
        // the column aggregate corresponds to the first value of the current buffer
        const aggregate_t columnAggregate = columnAggregates[0][(i + 1) % 2];

        // Write the computed aggregates back out to memory
        output[i] = {rowAggregate, columnAggregate};
    }
}
