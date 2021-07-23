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
        const DataPack readCompute(read.df, read.dg, read.inv);
        compute.write({readCompute, aggregate_t_init, 0});
    }

}

void ProcessingElement(const int stage, 
                       stream<InputDataPack, stream_d> &scatter_in,
                       stream<ComputePack, stream_d> &compute_in,
                       stream<aggregate_t, stream_d> &reduce_in,
                       stream<InputDataPack, stream_d> &scatter_out,
                       stream<ComputePack, stream_d> &compute_out,
                       stream<aggregate_t, stream_d> &reduce_out) {
    DataPack column[t];
    aggregate_t columnAggregate[t];
    data_t QT[t];

    const index_t columnBounds = n - m + 1 - t * stage;

    MatrixProfileScatter:
    for (index_t i = 0; i < n - m + 1 - t * stage; ++i) {
        #pragma HLS PIPELINE II=1
        const InputDataPack read = scatter_in.read();
        const DataPack compute(read.df, read.dg, read.inv);
        if (i < t) {
            QT[i] = read.QT;
            column[i] = compute;
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
        const data_t QTbackward = read.QTforward;
        aggregate_t aggregateBackward = read.aggregate;

        MatrixProfileTile:
        for (index_t j = 0; j < t; ++j) {
            const DataPack col = (j < columnBounds) 
                ? column[j] : DataPack(0);
            
            QT[j] += row.df * col.dg + col.df * row.dg;
            const bool inBounds = i <= stage * t + j - (m / 4);

            const data_t P = inBounds ? QT[j] * row.inv * col.inv : 0;

            const aggregate_t prevColumn = (i > 0) ? columnAggregate[j] : aggregate_t_init;
            columnAggregate[j] = (P > prevColumn.value) ? aggregate_t{P, i} : prevColumn;

            aggregate_t prevRow = (j < 16) ? aggregateBackward : rowReduce[i % 8][j % 16];
	        rowReduce[i % 8][j % 16] = P > prevRow.value ? aggregate_t(P, stage * t + j) : prevRow;
	    }

	    aggregate_t rowAggregate = TreeReduce::Maximum<aggregate_t, 16>(rowReduce[i % 8]);

        // shift values in QT forward
        data_t QTforward = QT[t - 1];
        MatrixProfileShiftQT:
        for (index_t j = t - 1; j > 0; --j) {
            #pragma HLS PIPELINE II=1
            QT[j] = QT[j - 1];
        }
        QT[0] = QTbackward;

        compute_out.write({row, rowAggregate, QTforward});
    }
    
    const int loopCount = t * (stage + 1) > (n - m + 1) ? (n - m + 1): (t * (stage + 1));
    MatrixProfileReduce:
    for (index_t i = 0; i < loopCount; ++i) {
        #pragma HLS PIPELINE II=1
        aggregate_t read = (i >= t * stage) 
                ? columnAggregate[i - t * stage] : reduce_in.read();
        reduce_out.write(read);
    }

}

void StreamToMemory(stream<ComputePack, stream_d> &compute, 
                    stream<aggregate_t, stream_d> &reduce,
                    data_t *MP, index_t *MPI) {
    aggregate_t rowAggregates[n - m + 1];

    StreamToMemoryReduceRow:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        const ComputePack read = compute.read();
        rowAggregates[i] = read.aggregate;
    }

    StreamToMemoryReduce:
    for (int i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        const aggregate_t rowAggregate = rowAggregates[i];
        const aggregate_t columnAggregate = reduce.read();
        const aggregate_t aggregate = rowAggregate.value > columnAggregate.value ? rowAggregate : columnAggregate;
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
    stream<aggregate_t, stream_d> reduce[nPE + 1];

    MemoryToStream(in, scatter[0], compute[0]);

    for (index_t i = 0; i < nPE; ++i) {
        #pragma HLS UNROLL
        ProcessingElement(i, scatter[i], compute[i], reduce[i],
                          scatter[i + 1], compute[i + 1], reduce[i + 1]);
    }

    StreamToMemory(compute[nPE], reduce[nPE], MP, MPI);
}