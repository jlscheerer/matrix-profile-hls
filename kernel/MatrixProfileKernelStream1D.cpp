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

static constexpr size_t stream_d = 3;

struct ScatterPack {
    data_t QT;
    ComputePack data;
    ScatterPack() = default;
    ScatterPack(data_t QT, ComputePack data)
        : QT(QT), data(data) {}
};

struct ComputationPack {
    ComputePack row;
    aggregate_t aggregate;
    data_t QTforward;
    ComputationPack() = default;
    ComputationPack(ComputePack row, aggregate_t aggregate, data_t QTforward)
        : row(row), aggregate(aggregate), QTforward(QTforward) {}
};

void MemoryToStream(const data_t *QT, const ComputePack *data, 
                    stream<ScatterPack, stream_d> &scatter, stream<ComputationPack, stream_d> &compute) {

    for (index_t i = 0; i < n - m + 1; ++i) {
        scatter.write(ScatterPack{QT[i], data[i]});
    }

    for (index_t i = 0; i < n - m + 1; ++i) {
        compute.write(ComputationPack{data[i], aggregate_t_init, 0});
    }

}

void ProcessingElement(const int stage, stream<ScatterPack, stream_d> &scatter_in,
        stream<ComputationPack, stream_d> &compute_in, stream<aggregate_t, stream_d> &reduce_in, stream<ScatterPack, stream_d> &scatter_out,
        stream<ComputationPack, stream_d> &compute_out, stream<aggregate_t, stream_d> &reduce_out) {

    ComputePack column[t];
    aggregate_t columnAggregate[t];
    data_t QT[t], P[t];

    // TODO: Change to implicit initiation
    for (index_t i = 0; i < t; ++i) {
        columnAggregate[i] = aggregate_t_init;
        QT[i] = 0;
        P[i] = 0;
        column[i] = {0, 0, 0};
    }

    for (index_t i = 0; i < n - m + 1 - t * stage; ++i) {
        ScatterPack read = scatter_in.read();
        if (i < t) {
            QT[i] = read.QT;
            column[i] = read.data;
        } else scatter_out.write(read);
    }

    for (index_t i = 0; i < n - m + 1; ++i) {
        const ComputationPack read = compute_in.read();

        const ComputePack row = read.row;
        const data_t QTbackward = read.QTforward;
        aggregate_t rowAggregate = read.aggregate;

        for (index_t j = 0; j < t; ++j) {
            QT[j] += row.df * column[j].dg + column[j].df * row.dg;
            const bool inBounds = i <= stage * t + j - (m / 4);

            P[j] = inBounds ? QT[j] * row.inv * column[j].inv : 0;
            
            columnAggregate[j] = (P[j] > columnAggregate[j].value) ? aggregate_t{P[j], i} : columnAggregate[j];
            rowAggregate = P[j] > rowAggregate.value ? aggregate_t{P[j], stage * t + j} : rowAggregate;
        }

        // Do TreeReduction on P for row Aggregate

        // shift values in QTforward
        data_t QTforward = QT[t - 1];
        for (index_t j = t - 1; j > 0; --j) {
            QT[j] = QT[j - 1];
        }

        QT[0] = QTbackward;
        compute_out.write(ComputationPack{row, rowAggregate, QTforward});
    }
    
    
    const int loopCount = t * (stage + 1) > (n - m + 1) ? (n - m + 1): (t * (stage + 1));
    for (index_t i = 0; i < loopCount; ++i) {
        aggregate_t read = (i >= t * stage) 
                ? columnAggregate[i - t * stage] : reduce_in.read();
        reduce_out.write(read);
    }
}

void StreamToMemory(stream<ComputationPack, stream_d> &compute, 
                    stream<aggregate_t, stream_d> &reduce,
                    data_t *MP, index_t *MPI) {

    aggregate_t rowAggregates[n - m + 1];
    for (index_t i = 0; i < n - m + 1; ++i) {
        const ComputationPack read = compute.read();
        rowAggregates[i] = read.aggregate;
    }

    aggregate_t columnAggregates[n - m + 1];
    for (index_t i = 0; i < n - m + 1; ++i) {
        const aggregate_t read = reduce.read();
        columnAggregates[i] = read;
    }

    for (int i = 0; i < n - m + 1; ++i) {
        const aggregate_t rowAggregate = rowAggregates[i];
        const aggregate_t columnAggregate = columnAggregates[i];
        const aggregate_t aggregate = rowAggregate.value > columnAggregate.value ? rowAggregate : columnAggregate;
        MP[i] = aggregate.value;
        MPI[i] = aggregate.index;
    }
}

void MatrixProfileKernelTLF(const data_t *QTInit, const ComputePack *data, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=QTInit offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MP     offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=MPI    offset=slave bundle=gmem3

    constexpr index_t nPE = (n - m + t) / t;

    stream<ScatterPack, stream_d> scatter[nPE + 1];
    stream<ComputationPack, stream_d> compute[nPE + 1];
    stream<aggregate_t, stream_d> reduce[nPE + 1];

    MemoryToStream(QTInit, data, scatter[0], compute[0]);

    for (index_t i = 0; i < nPE; ++i) {
        ProcessingElement(i, scatter[i], compute[i], reduce[i], scatter[i + 1], compute[i + 1], reduce[i + 1]);
    }

    StreamToMemory(compute[nPE], reduce[nPE], MP, MPI);
}