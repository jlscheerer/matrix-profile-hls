/**
 * @file    MatrixProfileKernelStream1D.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Stream-1D]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"
    #include "hls_math.h"
#endif

#include "kernel/Stream.hpp"
#include "kernel/TreeReduce.hpp"
#include "kernel/DataPacks.hpp"

void MemoryToStream(const index_t n, const index_t m, const index_t iteration,
                    const InputDataPack *columns,
                    Stream<InputDataPack> &scatter, Stream<ComputePack> &compute) {
    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    MemoryToStreamScatter:
    for (index_t i = 0; i < nColumns; ++i) {
        #pragma HLS PIPELINE II=1

        const index_t columnIndex = nOffset + i;
        const InputDataPack read = (columnIndex < n - m + 1) ? columns[columnIndex]
                                                             : InputDataPack(0);
        scatter.write(read);
    }

    MemoryToStreamCompute:
    for (index_t i = 0; i < nRows; ++i) {
        #pragma HLS PIPELINE II=1

        // rows will by design always by in bounds!
        const InputDataPack readRow = columns[i];
        const DataPack rowData(readRow.df, readRow.dg, readRow.inv);
        const aggregate_t rowAggregate = aggregate_t_init;

        const index_t columnIndex = nOffset + nColumns + i;
        const bool inBounds = columnIndex < n - m + 1;

        const InputDataPack readColumn = inBounds ? columns[columnIndex]
                                                  : InputDataPack(0);
        const DataPack columnData(readColumn.df, readColumn.dg, readColumn.inv);
        const aggregate_t columnAggregate = aggregate_t_init;

        compute.write({rowData, rowAggregate, columnData, columnAggregate});
    }

}

void ProcessingElement(const index_t n, const index_t m,
                       const index_t iteration, const index_t stage,
                       Stream<InputDataPack> &scatter_in,
                       Stream<ComputePack> &compute_in,
                       Stream<InputDataPack> &scatter_out,
                       Stream<ComputePack> &compute_out) {
    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    const index_t revStage = (nColumns - 1) / t - stage;

    DataPack columns[t];
    aggregate_t columnAggregates[t];
    data_t QT[t];

    const index_t afterMe = t * revStage;
    const index_t myCount = (stage == 0) ? (nColumns - revStage * t) : t;
    const index_t loopCount = afterMe + myCount;

    MatrixProfileScatter:
    for (index_t i = 0; i < loopCount; ++i) {
        #pragma HLS PIPELINE II=1

        InputDataPack read = scatter_in.read();
        if (i >= afterMe) {
            QT[i - afterMe] = read.QT;
            columns[i - afterMe] = {read.df, read.dg, read.inv};
        } else scatter_out.write(read);
    }

    // TODO: Comment w >= rowReduceD2
    constexpr int rowReduceD1 = 8;
    constexpr int rowReduceD2 = 8;
    aggregate_t rowReduce[rowReduceD1][rowReduceD2];
    #pragma HLS ARRAY_PARTITION variable=rowReduce dim=2 complete

    MatrixProfileCompute:
    for (index_t i = 0; i < nRows; ++i) {
        const ComputePack read = compute_in.read();

        const DataPack row = read.row;
        const aggregate_t rowAggregateBackward = read.rowAggregate;

        const DataPack columnBackward = read.column;
        const aggregate_t columnAggregateBackward = read.columnAggregate;

        MatrixProfileTile:
        for (index_t j = 0; j < t; ++j) {
            #pragma HLS PIPELINE II=1

            const DataPack column = columns[j];

            QT[j] += row.df * column.dg + column.df * row.dg;

            const index_t rowIndex = i;
            const index_t columnIndex = nOffset + afterMe + i + j;

            const bool columnInBounds = columnIndex < n - m + 1;
            const bool exclusionZone = rowIndex > columnIndex - m / 4;
            const bool inBounds = columnInBounds && !exclusionZone;

            const data_t P = inBounds ? static_cast<data_t>(QT[j] * row.inv * column.inv) : static_cast<data_t>(0);

            const aggregate_t prevRow = (j < rowReduceD2) ? rowAggregateBackward
                                                          : rowReduce[i % rowReduceD1][j % rowReduceD2];
            rowReduce[i % rowReduceD1][j % rowReduceD2] = prevRow.value > P ? prevRow
                                                                            : aggregate_t(P, columnIndex);

            const aggregate_t prevColumn = (i > 0) ? columnAggregates[j]
                                                   : aggregate_t_init;
            columnAggregates[j] = (prevColumn.value > P) ? prevColumn
                                                         : aggregate_t(P, rowIndex);
        }

        const aggregate_t rowAggregate = TreeReduce::Maximum<aggregate_t, rowReduceD2>(rowReduce[i % rowReduceD1]);

        const DataPack columnForward = columns[0];
        const aggregate_t columnAggregateForward = columnAggregates[0];

        // shift the current column to the right (i.e. backwards)
        MatrixProfileShift:
        for (index_t j = 0; j < t - 1; ++j) {
            #pragma HLS PIPELINE II=1

            columns[j] = columns[j + 1];
            columnAggregates[j] = columnAggregates[j + 1];
        }
        columns[t - 1] = columnBackward;
        columnAggregates[t - 1] = columnAggregateBackward;

        // Propagate Values along the pipeline
        compute_out.write({row, rowAggregate, columnForward, columnAggregateForward});
    }
}

void StreamToMemory(const index_t n, const index_t m, const index_t iteration,
                    Stream<ComputePack> &compute, OutputDataPack *result) {
    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    StreamToMemoryReduce:
    for (index_t i = 0; i < nRows; ++i) {
        #pragma HLS PIPELINE II=1

        const ComputePack read = compute.read();

        const aggregate_t rowAggregate = read.rowAggregate;
        const aggregate_t columnAggregate = read.columnAggregate;

        result[i] = {rowAggregate, columnAggregate};
    }
}

void MatrixProfileKernelTLF(const index_t n, const index_t m, const index_t iteration,
                            const InputDataPack *input, OutputDataPack *output) {
    #pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

    #pragma HLS DATAFLOW

    constexpr index_t nPE = (nColumns + t - 1) / t;

    Stream<InputDataPack> scatter[nPE + 1];
    Stream<ComputePack> compute[nPE + 1];

    MemoryToStream(n, m, iteration, input, scatter[0], compute[0]);

    for (index_t i = 0; i < nPE; ++i) {
        #pragma HLS UNROLL

        ProcessingElement(n, m, iteration, i, scatter[i], compute[i],
                          scatter[i + 1], compute[i + 1]);
    }

    StreamToMemory(n, m, iteration, compute[nPE], output);
}