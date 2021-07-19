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
void MemoryToStreamElement(const data_t *QT, const ComputePack *data, stream<data_t, stream_d> &QT_s, stream<ComputePack, stream_d> &data_s,
                           stream<aggregate_t,  stream_d> &scatterLane, stream<aggregate_t, stream_d> &reductionLane) {
    // =============== [Scatter] ===============
    for (index_t i = 0; i < n - m + 1; ++i) {
        QT_s.write(QT[i]);
        data_s.write(data[i]);
        scatterLane.write(aggregate_t_init);
    }
    // =============== [/Scatter] ===============

    // =============== [Reduce] ===============
    PrecomputationInitReduce:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        reductionLane.write(aggregate_t_init);
    }
    // =============== [/Reduce] ===============
}

void DiagonalComputeElement(const index_t stage, stream<data_t, stream_d> &QT_in, stream<ComputePack, stream_d> &data_in,
                            stream<aggregate_t, stream_d> &scatterLane_in, stream<aggregate_t, stream_d> &reductionLane_in, stream<data_t, stream_d> &QT_out, 
                            stream<ComputePack, stream_d> &data_out, stream<aggregate_t, stream_d> &scatterLane_out, stream<aggregate_t, stream_d> &reductionLane_out) {
    // local "cache" [size: (n - m + 1 - stage) would be sufficient]
    ComputePack rowData[n - m + 1], columnData[n - m + 1];
    
    data_t QT[t], P[t];
    aggregate_t rowAggregate[n - m + 1], columnAggregate[n - m + 1];

    // =============== [Scatter] ===============
    MatrixProfileScatter:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1

        data_t QTforward = QT_in.read();
        ComputePack data = data_in.read();
        aggregate_t aggregate = scatterLane_in.read();

        // store values in local "cache"
        if (k >= t * stage && k < t * (stage + 1))
            QT[k - t * stage] = QTforward;
        
        rowData[k] = data; columnData[k] = data;
        rowAggregate[k] = aggregate; columnAggregate[k] = aggregate;

        // forward values to subsequent processing elements
        QT_out.write(QTforward);
        data_out.write(data);
        scatterLane_out.write(aggregate);
    }
    // =============== [/Scatter] ===============

    // =============== [Compute] ===============

    MatrixProfileCompute:
    for (index_t k = 0; k < n - m + 1; ++k) {
        for (index_t i = 0; i < t; ++i) {
            #pragma HLS PIPELINE II=1
            const ComputePack row = rowData[k];
            const bool computationInRange = stage * t + k + i < n - m + 1;
            const ComputePack column = computationInRange ? columnData[stage * t + k + i] : (ComputePack) {0, 0, 0};

            QT[i] += row.df * column.dg + column.df * row.dg;
            P[i] = QT[i] * row.inv * column.inv;

            const bool exclusionZone = stage * t + i < m / 4;

            if (computationInRange && !exclusionZone) {
                if (P[i] > rowAggregate[k].value)
                    rowAggregate[k] = (aggregate_t){P[i], stage * t + k + i};
                if (P[i] > columnAggregate[stage * t + k + i].value)
                    columnAggregate[stage * t + k + i] = (aggregate_t){P[i], k};
            }
        }
    }

    // =============== [/Compute] ===============

    // =============== [Reduce] ===============
    MatrixProfileReduce:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1
        // get previous aggregate from predecessor
        aggregate_t prevAggregate = reductionLane_in.read();
        // merge row and column aggregates
        aggregate_t currAggregate = (columnAggregate[k].value > rowAggregate[k].value) 
                                    ? columnAggregate[k] : rowAggregate[k];
        if (currAggregate.value > prevAggregate.value)
            reductionLane_out.write(currAggregate);
        else reductionLane_out.write(prevAggregate);
    }
    // =============== [/Reduce] ===============
}

void StreamToMemoryElement(stream<data_t, stream_d> &QT, stream<ComputePack, stream_d> &data, stream<aggregate_t, stream_d> &scatterLane,
                           stream<aggregate_t, stream_d> &reductionLane, data_t *MP, index_t *MPI) {
    StreamToMemorySink:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1
        QT.read();
        data.read();
        scatterLane.read();
    }

    StreamToMemoryReduce:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1
        const aggregate_t aggregate = reductionLane.read();
        MP[k] = aggregate.value;
        MPI[k] = aggregate.index;
    }
}

void MatrixProfileKernelTLF(const data_t *QTInit, const ComputePack *data, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=QTInit offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=MP     offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI    offset=slave bundle=gmem2

    #pragma HLS DATAFLOW

    constexpr index_t numStages = (n - m + 1 + t - 1) / t;

    // Streams required to calculate Correlations
    stream<data_t, stream_d> QT_s[numStages + 1];
    stream<ComputePack, stream_d> data_s[numStages + 1];

    // Store the intermediate results
    stream<aggregate_t, stream_d> scatterLane[numStages + 1], reductionLane[numStages + 1];

    MemoryToStreamElement(QTInit, data, QT_s[0], data_s[0], scatterLane[0], reductionLane[0]);

    for (index_t stage = 0; stage < numStages; ++stage) {
        #pragma HLS UNROLL
        DiagonalComputeElement(stage, QT_s[stage], data_s[stage], scatterLane[stage], reductionLane[stage], 
                               QT_s[stage + 1], data_s[stage + 1], scatterLane[stage + 1], reductionLane[stage + 1]);
    }

    StreamToMemoryElement(QT_s[numStages], data_s[numStages], scatterLane[numStages], reductionLane[numStages], MP, MPI);
}
