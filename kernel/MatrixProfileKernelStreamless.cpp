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

void MatrixProfileKernelTLF(const index_t n, const index_t m, const index_t iteration,
                            const InputDataPack *columns, const InputDataPack *rows, OutputDataPack *result) {
    #pragma HLS INTERFACE m_axi port=columns offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=rows    offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=result  offset=slave bundle=gmem2

    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    // TODO: Comment about QT
    data_t QT[nColumns];

    constexpr int rowReduceD1 = 8;
    constexpr int rowReduceD2 = 8;
    aggregate_t rowReduce[rowReduceD1][rowReduceD2];
    #pragma HLS ARRAY_PARTITION variable=rowReduce dim=2 complete

#if 0
    // TODO: Only required for t <= rowReduceD2
    MatrixProfileInitReduce:
    for (index_t i = 0; i < rowReduceD1; ++i) {
        for (index_t j = 0; j < rowReduceD2; ++j) {
            #pragma HLS UNROLL
            rowReduce[i][j] = aggregate_t_init;
        }
    }
#endif

    // TODO: Double-Buffer comment
    aggregate_t columnAggregates[nColumns][2];

    MatrixProfileInitQT:
    for (index_t i = 0; i < nColumns; ++i) {
        const index_t columnIndex = nOffset + i;
        const bool inBounds = columnIndex < n - m + 1;
        const InputDataPack read = inBounds ? columns[columnIndex]
                                            : InputDataPack(0);
        QT[i] = read.QT;
    }
	
    MatrixProfileComputeRow:
    for (index_t i = 0; i < nRows; ++i) {
        const index_t rowIndex = i;
        const InputDataPack row = rows[rowIndex];

        MatrixProfileComputeTile:
        for (index_t j = 0; j < nColumns; ++j) {
            const index_t columnIndex = nOffset + i + j;

            const bool columnInBounds = columnIndex < n - m + 1;
            const bool exclusionZone = rowIndex > columnIndex - m / 4;
            const bool inBounds = columnInBounds && !exclusionZone;

            const InputDataPack column = columnInBounds ? columns[columnIndex]
                                                        : InputDataPack(0);

            QT[j] += row.df * column.dg + column.df * row.dg;
            const data_t P = inBounds ? QT[j] * row.inv * column.inv : 0;

            aggregate_t prevRow = (j < rowReduceD2) ? aggregate_t_init : rowReduce[i % rowReduceD1][j % rowReduceD2];
	        rowReduce[i % rowReduceD1][j % rowReduceD2] = prevRow.value > P ? prevRow : aggregate_t(P, columnIndex);

            const aggregate_t prevColumn = (j < nColumns - 1 && i > 0) ? columnAggregates[j + 1][i % 2] 
                                                                       : aggregate_t_init;
            columnAggregates[j][(i + 1) % 2] = P > prevColumn.value ? aggregate_t(P, rowIndex) 
                                                                    : prevColumn;
        }

        const aggregate_t rowAggregate = TreeReduce::Maximum<aggregate_t, rowReduceD2>(rowReduce[i % rowReduceD1]);
        const aggregate_t columnAggregate = columnAggregates[0][(i + 1) % 2];

        result[i] = {rowAggregate, columnAggregate};
    }
}

