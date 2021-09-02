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

    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    // TODO: Comment about QT
    data_t QT[nColumns];

    // TODO: Comment w >= rowReduceD2
    constexpr int rowReduceD1 = 8;
    constexpr int rowReduceD2 = 8;
    aggregate_t rowReduce[rowReduceD1][rowReduceD2];
    #pragma HLS ARRAY_PARTITION variable=rowReduce dim=2 complete

    // TODO: Double-Buffer comment
    aggregate_t columnAggregates[nColumns][2];

    MatrixProfileInitQT:
    for (index_t i = 0; i < nColumns; ++i) {
	#pragma HLS PIPELINE II=1

        const index_t columnIndex = nOffset + i;
        const bool inBounds = columnIndex < n - m + 1;
        const InputDataPack read = inBounds ? input[columnIndex]
                                            : InputDataPack(0);
        QT[i] = read.QT;
    }
	
    MatrixProfileComputeRow:
    for (index_t i = 0; i < nRows; ++i) {
        const index_t rowIndex = i;
        const InputDataPack row = input[rowIndex];

        MatrixProfileComputeTile:
        for (index_t j = 0; j < nColumns; ++j) {
	    #pragma HLS PIPELINE II=1

            const index_t columnIndex = nOffset + i + j;

            const bool columnInBounds = columnIndex < n - m + 1;
            const bool exclusionZone = rowIndex > columnIndex - m / 4;
            const bool inBounds = columnInBounds && !exclusionZone;

            const InputDataPack column = columnInBounds ? input[columnIndex]
                                                        : InputDataPack(0);

            QT[j] += row.df * column.dg + column.df * row.dg;
            const data_t P = inBounds ? static_cast<data_t>(QT[j] * row.inv * column.inv) : static_cast<data_t>(0);

            const aggregate_t prevRow = (j < rowReduceD2) ? aggregate_t_init 
                                                          : rowReduce[i % rowReduceD1][j % rowReduceD2];
            rowReduce[i % rowReduceD1][j % rowReduceD2] = prevRow.value > P ? prevRow 
                                                                            : aggregate_t(P, columnIndex);

            const aggregate_t prevColumn = (j < nColumns - 1 && i > 0) ? columnAggregates[j + 1][i % 2] 
                                                                       : aggregate_t_init;
            columnAggregates[j][(i + 1) % 2] = P > prevColumn.value ? aggregate_t(P, rowIndex) 
                                                                    : prevColumn;
        }

        const aggregate_t rowAggregate = TreeReduce::Maximum<aggregate_t, rowReduceD2>(rowReduce[i % rowReduceD1]);
        const aggregate_t columnAggregate = columnAggregates[0][(i + 1) % 2];

        output[i] = {rowAggregate, columnAggregate};
    }
}
