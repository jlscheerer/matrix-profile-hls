/**
 * @file    MatrixProfileKernelStream1D.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Stream-1D]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    
    #include "kernel/MatrixProfileKernel.hpp"
    #include "kernel/TreeReduce.hpp"
    
    #include "hls_math.h"
    #include "hls_stream.h"
    using hls::stream;
#endif

#include "kernel/DataPacks.hpp"
static constexpr size_t stream_d = 3;

void MemoryToStream(const data_t *T,
                    stream<ScatterPack, stream_d> &scatter,
                    stream<ComputePack, stream_d> &compute) {
    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];
    #pragma HLS ARRAY_PARTITION variable=T_m complete

    // the first m T values, required for convolution
    data_t Ti_m[m];
    #pragma HLS ARRAY_PARTITION variable=Ti_m complete

    // store ComputePacks for rows to propagate later
    DataPack rowData[n - m + 1];

    data_t mu0 = 0, inv_sum = 0, qt_sum = 0;
    MemoryToStreamInitTMu:
    for (index_t i = 0; i < m; ++i) {
        data_t T_i = T[i];
        mu0 += T_i;
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }
    mu0 /= m;

    MemoryToStreamInitInvQT:
    for (index_t k = 0; k < m; ++k) {
        inv_sum += (T_m[k] - mu0) * (T_m[k] - mu0);
        qt_sum += (T_m[k] - mu0) * (Ti_m[k] - mu0);
    }
    data_t inv0 = static_cast<data_t>(1) / sqrt(inv_sum);

    rowData[0] = {0, 0, inv0};
    scatter.write({qt_sum, 0, 0, inv0});

    MemoryToStreamPrecomputeScatter:
    for (index_t i = m; i < n; ++i) {
        data_t T_i = T[i];
        data_t T_r = T_m[0];

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

        rowData[i - m + 1] = {df, dg, inv};
        scatter.write({qt_sum, df, dg, inv});

        // shift all values in T_m back
        PrecomputationComputeShift: 
        for (index_t k = 0; k < m - 1; ++k)
            T_m[k] = T_m[k + 1];
        T_m[m - 1] = T_i;
    }
    // =============== [/Precompute] ===============
    
    MemoryToStreamCompute:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        const DataPack read = rowData[i];
        const DataPack readCompute(read.df, read.dg, read.inv);
        compute.write({readCompute, aggregate_t_init, 0});
    }
}

void ProcessingElement(const int stage, 
                       stream<ScatterPack, stream_d> &scatter_in,
                       stream<ComputePack, stream_d> &compute_in,
                       stream<aggregate_t, stream_d> &reduce_in,
                       stream<ScatterPack, stream_d> &scatter_out,
                       stream<ComputePack, stream_d> &compute_out,
                       stream<aggregate_t, stream_d> &reduce_out) {
    DataPack column[t];
    aggregate_t columnAggregate[t];
    data_t QT[t];

    const index_t columnBounds = n - m + 1 - t * stage;

    MatrixProfileScatter:
    for (index_t i = 0; i < n - m + 1 - t * stage; ++i) {
        #pragma HLS PIPELINE II=1
        const ScatterPack read = scatter_in.read();
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

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    #pragma HLS INLINE
    return sqrt(static_cast<data_t>(2 * m * (1 - PearsonCorrelation)));
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
        MP[i] = PearsonCorrelationToEuclideanDistance(aggregate.value);
        MPI[i] = aggregate.index;
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem2

    #pragma HLS DATAFLOW

    constexpr index_t nPE = (n - m + t) / t;

    stream<ScatterPack, stream_d> scatter[nPE + 1];
    stream<ComputePack, stream_d> compute[nPE + 1];
    stream<aggregate_t, stream_d> reduce[nPE + 1];

    MemoryToStream(T, scatter[0], compute[0]);

    for (index_t i = 0; i < nPE; ++i) {
        #pragma HLS UNROLL
        ProcessingElement(i, scatter[i], compute[i], reduce[i],
                          scatter[i + 1], compute[i + 1], reduce[i + 1]);
    }

    StreamToMemory(compute[nPE], reduce[nPE], MP, MPI);
}
