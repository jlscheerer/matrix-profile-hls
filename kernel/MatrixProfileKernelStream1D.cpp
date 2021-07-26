/**
 * @file    MatrixProfileKernelStream1D.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Stream-1D]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"
    
    #include "hls_math.h"
    #include "hls_stream.h"
    using hls::stream;
#endif

#include "kernel/DataPacks.hpp"
#include "kernel/TreeReduce.hpp"

static constexpr size_t stream_d = 3;

void MemoryToStream(const InputDataPack *in, 
                    stream<InputDataPack, stream_d> &scatter,
                    stream<ComputePack, stream_d> &compute) {
    MemoryToStreamScatter:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        const InputDataPack read = in[i];
        scatter.write(read);
    }
    
    MemoryToStreamCompute:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        const InputDataPack read = in[i];
        const DataPack row(read.df, read.dg, read.inv);
        compute.write({row, aggregate_t_init, DataPack(), aggregate_t_init});
    }

}

void ProcessingElement(const int stage, 
                       stream<InputDataPack, stream_d> &scatter_in,
                       stream<ComputePack, stream_d> &compute_in,
                       stream<InputDataPack, stream_d> &scatter_out,
                       stream<ComputePack, stream_d> &compute_out) {
    const index_t revStage = (n - m) / t - stage;

    DataPack columns[t];
    aggregate_t columnAggregates[t];
    data_t QT[t];

    const int afterMe = t * revStage;
	const int myCount = (stage == 0) ? (n - m + 1 - revStage * t) : t;
	const int loopCount = afterMe + myCount;

    MatrixProfileScatter:
    for (index_t i = 0; i < loopCount; ++i) {
        #pragma HLS PIPELINE II=1
        InputDataPack read = scatter_in.read();
        if (i >= afterMe) {
            QT[i - afterMe] = read.QT;
            columns[i - afterMe] = {read.df, read.dg, read.inv};
        } else scatter_out.write(read);
    }

    aggregate_t rowReduce[8][16];
    #pragma HLS ARRAY_PARTITION variable=rowReduce dim=2 complete

    // TODO: Only required for t <= 16
    MatrixProfileInitReduce:
    for (index_t i = 0; i < 8; ++i) {
        for (index_t j = 0; j < 16; ++j) {
            #pragma HLS UNROLL
            rowReduce[i][j] = aggregate_t_init;
        }
    }

    MatrixProfileCompute:
    for (index_t i = 0; i < n - m + 1; ++i) {
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
            const index_t columnIndex = afterMe + i + j;

            const bool columnInBounds = columnIndex < n - m + 1;
            const bool exclusionZone = rowIndex > columnIndex - m / 4;
            const bool inBounds = columnInBounds && !exclusionZone;

            const data_t P = inBounds ? QT[j] * row.inv * column.inv : 0;

            aggregate_t prevRow = (j < 16) ? rowAggregateBackward : rowReduce[i % 8][j % 16];
	        rowReduce[i % 8][j % 16] = prevRow.value > P ? prevRow : aggregate_t(P, columnIndex);

            const aggregate_t prevColumn = (i > 0) ? columnAggregates[j] : aggregate_t_init;
            columnAggregates[j] = (prevColumn.value > P) ? prevColumn : aggregate_t(P, rowIndex);
        }

	    const aggregate_t rowAggregate = TreeReduce::Maximum<aggregate_t, 16>(rowReduce[i % 8]);

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

void StreamToMemory(stream<ComputePack, stream_d> &compute,
                    data_t *MP, index_t *MPI) {
    StreamToMemoryReduce:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        const ComputePack read = compute.read();

        const aggregate_t rowAggregate = read.rowAggregate;
        const aggregate_t columnAggregate = read.columnAggregate;
        
        const aggregate_t aggregate = rowAggregate.value > columnAggregate.value 
						? rowAggregate : columnAggregate;
        
        MP[i] = aggregate.value;
        MPI[i] = aggregate.index;
    }
}

void MatrixProfileKernelTLF(const InputDataPack *in,
                            data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem1

    #pragma HLS DATAFLOW

    constexpr index_t nPE = (n - m + t) / t;

    stream<InputDataPack, stream_d> scatter[nPE + 1];
    stream<ComputePack, stream_d> compute[nPE + 1];

    MemoryToStream(in, scatter[0], compute[0]);

    for (index_t i = 0; i < nPE; ++i) {
        #pragma HLS UNROLL
        ProcessingElement(i, scatter[i], compute[i], 
                             scatter[i + 1], compute[i + 1]);
    }

    StreamToMemory(compute[nPE], MP, MPI);
}