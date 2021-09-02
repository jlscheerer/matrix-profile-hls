/**
 * @file    MatrixProfileKernelTiled.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Tiled-Kernel]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"
    #include "hls_math.h"
#endif

#include "kernel/Stream.hpp"
#include "kernel/TreeReduce.hpp"
#include "kernel/DataPacks.hpp"

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"

    #include "hls_math.h"
#endif

#include "kernel/Stream.hpp"
#include "kernel/TreeReduce.hpp"
#include "kernel/DataPacks.hpp"

// The Tiled Kernel realized a systolic array architecture.
// As described in the thesis DiagonalComputeElements are 
// connected in reverse:
/*
┌──────────────────────┐      ┌──────────────────────┐                 ┌──────────────────────┐      ┌──────────────────────┐
│    StreamToMemory    ◀──────┤DiagonalComputeElement◀───────...◀──────┤DiagonalComputeElement◀──────┤    MemoryToStream    │
└──────────────────────┘      └──────────────────────┘                 └──────────────────────┘      └──────────────────────┘
*/

// Processing Element responsible for reading input data provided by the Driver Application.
void MemoryToStream(const index_t n, const index_t m, const index_t iteration,
                    const InputDataPack *input, Stream<InputDataPack> &scatter, 
                    Stream<ComputePack> &compute) {
    // Determine values corresponding to the current iteration
    // In particular, the number of rows of the current DiagonalChunk
    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    // First Scatter Data corresponding to columns of the DiagonalChunk
    MemoryToStreamScatter:
    for (index_t i = 0; i < nColumns; ++i) {
        #pragma HLS PIPELINE II=1
        
        // In case the DiagonalChunk exceeds the size of the input (columnIndex >= n - m + 1),
        // we propagate 0s as not to affect the result
        const index_t columnIndex = nOffset + i;
        const InputDataPack read = (columnIndex < n - m + 1) ? input[columnIndex]
                                                             : InputDataPack(0);
        // Propagte values to the first DiagonalComputeElement via scatter[0]
        scatter.write(read);
    }

    // Scatter Phase during Computation
    MemoryToStreamCompute:
    for (index_t i = 0; i < nRows; ++i) {
        // As we need to access the input buffer twice, we require II=2.
        // However, the computation performed by the DiagonalComputeElements
        // is more latent; therefore, this has no practical effect on performance.
        #pragma HLS PIPELINE II=2

        // Read values corresponding to the i-th row.
        // Each row is by design allows in-bounds.
        const InputDataPack readRow = input[i];
        const DataPack rowData(readRow.df, readRow.dg, readRow.inv);
        const aggregate_t rowAggregate = aggregate_t_init;

        // Determine the index of the new column value, i.e., the new value 
        // required by the first DiagonalComputeElement
        const index_t columnIndex = nOffset + nColumns + i;
        const bool inBounds = columnIndex < n - m + 1;

        // Propagate the . In case the columnIndex is not in bounds, we 
        // propagate 0. This introduces the described extraneous triangle.
        const InputDataPack readColumn = inBounds ? input[columnIndex]
                                                  : InputDataPack(0);
        const DataPack columnData(readColumn.df, readColumn.dg, readColumn.inv);
        const aggregate_t columnAggregate = aggregate_t_init;

        // Propagate the current row, the new column value, and "dummy" aggregates
        compute.write({rowData, rowAggregate, columnData, columnAggregate});
    }

}

// DiagonalComputeElement: Compute a set of t-diagonals of the current DiagonalChunk
void DiagonalComputeElement(const index_t n, const index_t m,
                            const index_t iteration, const index_t stage,
                            Stream<InputDataPack> &scatter_in,
                            Stream<ComputePack> &compute_in,
                            Stream<InputDataPack> &scatter_out,
                            Stream<ComputePack> &compute_out) {
    // Determine values corresponding to the current iteration
    // In particular, the number of rows of the current DiagonalChunk
    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    // Determine the "reverse" stage, i.e. map the first Processing 
    // Element to the greatest index
    const index_t revStage = (nColumns - 1) / t - stage;

    // Buffer to contain the currently relevant column (aggregate) values (act shift registers)
    DataPack columns[t];
    aggregate_t columnAggregates[t];

    // QT values of the t-diagonals
    data_t QT[t];

    // Number of elements to forward during the explicit ScatterPhase
    const index_t afterMe = t * revStage;
    
    // Number of values required for the current ProcessingElement
    const index_t myCount = (stage == 0) ? (nColumns - revStage * t) : t;
    
    // How many elements to read during the explicit ScatterPhase
    const index_t loopCount = afterMe + myCount;

    MatrixProfileScatter:
    for (index_t i = 0; i < loopCount; ++i) {
        #pragma HLS PIPELINE II=1

        // Read column data from upstream neighbor
        InputDataPack read = scatter_in.read();

        // If the value is relevant for the current Processing Element,
        // store it in the internal buffers, otherwise propagate it downstream
        if (i >= afterMe) {
            QT[i - afterMe] = read.QT;
            columns[i - afterMe] = {read.df, read.dg, read.inv};
        } else scatter_out.write(read);
    }

    constexpr int rowReduceD1 = 8;
    constexpr int rowReduceD2 = 8;

    // Buffer used for partial reduction of rowAggregates
    // As opposed to the Thesis we use a two-dimensional buffer
    // to lower memory requirements of the TreeReduction
    aggregate_t rowReduce[rowReduceD1][rowReduceD2];
    #pragma HLS ARRAY_PARTITION variable=rowReduce dim=2 complete

    // Compute Unit: Main Computation
    MatrixProfileCompute:
    for (index_t i = 0; i < nRows; ++i) {
        // Read the ComputePack from the upstream neighbor
        const ComputePack read = compute_in.read();

        // Get the data corresponding to the current row
        const DataPack row = read.row;
        const aggregate_t rowAggregateBackward = read.rowAggregate;

        // Get column data from the upstream neighbor
        // Value needs to be integrated into internal buffer
        const DataPack columnBackward = read.column;
        const aggregate_t columnAggregateBackward = read.columnAggregate;

        // Go over all our t-diagonals 
        MatrixProfileTile:
        for (index_t j = 0; j < t; ++j) {
            #pragma HLS PIPELINE II=1

            // Get the column data for the j-th diagonal
            const DataPack column = columns[j];

            // Apply the SCAMP update formulation
            QT[j] += row.df * column.dg + column.df * row.dg;
            
            // Calculate corresponding row and column indices
            const index_t rowIndex = i;
            const index_t columnIndex = nOffset + afterMe + i + j;

            // Check whether the calculated values is relevant for the matrix 
            // profile, i.e., is both in bounds and not in the exclusion zone
            const bool columnInBounds = columnIndex < n - m + 1;
            const bool exclusionZone = rowIndex > columnIndex - m / 4;
            const bool inBounds = columnInBounds && !exclusionZone;

            // In case the computed value is not relevant take 0 (will not affect the result)
            const data_t P = inBounds ? static_cast<data_t>(QT[j] * row.inv * column.inv) : static_cast<data_t>(0);

            // Use partial reduction of rowAggregates to break loop dependency
            const aggregate_t prevRow = (j < rowReduceD2) ? rowAggregateBackward
                                                          : rowReduce[i % rowReduceD1][j % rowReduceD2];
            rowReduce[i % rowReduceD1][j % rowReduceD2] = prevRow.value > P ? prevRow
                                                                            : aggregate_t(P, columnIndex);

            // Implicit initialization of columnAggregates & simultaneous update
            const aggregate_t prevColumn = (i > 0) ? columnAggregates[j]
                                                   : aggregate_t_init;
            columnAggregates[j] = (prevColumn.value > P) ? prevColumn
                                                         : aggregate_t(P, rowIndex);
        }

        // determine the row-wise aggregate using TreeReduction
        const aggregate_t rowAggregate = TreeReduce::Maximum<aggregate_t, rowReduceD2>(rowReduce[i % rowReduceD1]);

        // values to forward to downstream neighbor
        const DataPack columnForward = columns[0];
        const aggregate_t columnAggregateForward = columnAggregates[0];

        // shift the current column data to the left (i.e. backwards)
        // (the "experimental" branch contains a version where this shift 
        //  is integrated into the main computation loop)
        MatrixProfileShift:
        for (index_t j = 0; j < t - 1; ++j) {
            #pragma HLS PIPELINE II=1

            columns[j] = columns[j + 1];
            columnAggregates[j] = columnAggregates[j + 1];
        }
        // integrate values from our downstream neighbor
        columns[t - 1] = columnBackward;
        columnAggregates[t - 1] = columnAggregateBackward;

        // Propagate values to our downstream neighbor
        compute_out.write({row, rowAggregate, columnForward, columnAggregateForward});
    }
}

// Processing Element responsible for writing the output data back out to memory.
void StreamToMemory(const index_t n, const index_t m, const index_t iteration,
                    Stream<ComputePack> &compute, OutputDataPack *result) {
    // Determine values corresponding to the current iteration
    // In particular, the number of rows of the current DiagonalChunk
    const index_t nOffset = iteration * nColumns;
    const index_t nRows = n - m + 1 - nOffset;

    StreamToMemoryReduce:
    for (index_t i = 0; i < nRows; ++i) {
        #pragma HLS PIPELINE II=1
        
        // Take values from the final DiagonalComputeElement
        const ComputePack read = compute.read();

        // Extract the corresponding row- and column-wise aggregates
        const aggregate_t rowAggregate = read.rowAggregate;
        const aggregate_t columnAggregate = read.columnAggregate;

        // Write these aggregates back out to memory
        result[i] = {rowAggregate, columnAggregate};
    }
}

void MatrixProfileKernelTLF(const index_t n, const index_t m, const index_t iteration,
                            const InputDataPack *input, OutputDataPack *output) {
    #pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

    #pragma HLS DATAFLOW

    // Determine the number of Processing Elements (DiagonalComputeElements)
    constexpr index_t nPE = (nColumns + t - 1) / t;

    // Streams used for the explicit Scatter Phase / during Computation
    Stream<InputDataPack> scatter[nPE + 1];
    Stream<ComputePack> compute[nPE + 1];

    // Read and propagate input data from global memory
    MemoryToStream(n, m, iteration, input, scatter[0], compute[0]);

    for (index_t i = 0; i < nPE; ++i) {
        #pragma HLS UNROLL

        // Processing Elements connected in sequence via Streams
        DiagonalComputeElement(n, m, iteration, i, scatter[i], compute[i],
                               scatter[i + 1], compute[i + 1]);
    }

    // Write the result (i.e., obtained aggregates) back to global memory
    StreamToMemory(n, m, iteration, compute[nPE], output);
}