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
void MemoryToStreamElement(const data_t *QT, const data_t *df, const data_t *dg, const data_t *inv, 
                           stream<data_t, stream_d> &QT_s, stream<data_t, stream_d> &df_s, stream<data_t, stream_d> &dg_s,
                           stream<data_t, stream_d> &inv_s,  stream<aggregate_t,  stream_d> &scatterLane, stream<aggregate_t, stream_d> &reductionLane) {
    // =============== [Scatter] ===============
    for (index_t i = 0; i < n - m + 1; ++i) {
        QT_s.write(QT[i]);
        df_s.write(df[i]);
        dg_s.write(dg[i]);
        inv_s.write(inv[i]);
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

void DiagonalComputeElement(const index_t stage, stream<data_t, stream_d> &QT_in, stream<data_t, stream_d> &df_in, stream<data_t, stream_d> &dg_in, 
                            stream<data_t, stream_d> &inv_in, stream<aggregate_t, stream_d> &scatterLane_in, stream<aggregate_t, stream_d> &reductionLane_in,
                            stream<data_t, stream_d> &QT_out, stream<data_t, stream_d> &df_out, stream<data_t, stream_d> &dg_out,
                            stream<data_t, stream_d> &inv_out, stream<aggregate_t, stream_d> &scatterLane_out, stream<aggregate_t, stream_d> &reductionLane_out) {
    // local "cache" [size: (n - m + 1 - stage) would be sufficient]
    data_t df_m[n - m + 1], dg_m[n - m + 1], inv_m[n - m + 1];
    
    data_t QT[t], P[t];
    aggregate_t rowAggregate[n - m + 1], columnAggregate[n - m + 1];

    // =============== [Scatter] ===============
    MatrixProfileScatter:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1

        data_t QTforward = QT_in.read();
        data_t dfi = df_in.read(), dgi = dg_in.read(), invi = inv_in.read();
        aggregate_t aggregate = scatterLane_in.read();

        // store values in local "cache"
        if (k >= t * stage && k < t * (stage + 1))
            QT[k - t * stage] = QTforward;
        
        df_m[k] = dfi; dg_m[k] = dgi; inv_m[k] = invi;
        rowAggregate[k] = aggregate;
        columnAggregate[k] = aggregate;

        // forward values to subsequent processing elements
        QT_out.write(QTforward);
        df_out.write(dfi); dg_out.write(dgi); inv_out.write(invi);
        scatterLane_out.write(aggregate);
    }
    // =============== [/Scatter] ===============

    // =============== [Compute] ===============

    MatrixProfileCompute:
    for (index_t k = 0; k < n - m + 1; ++k) {
        for (index_t i = 0; i < t; ++i) {
            #pragma HLS PIPELINE II=1

            const data_t dfi = df_m[k], dgi = dg_m[k], invi = inv_m[k];
            const bool computationInRange = stage * t + k + i < n - m + 1;
            const data_t dfj = computationInRange ? df_m[stage * t + k + i] : static_cast<data_t>(0);
            const data_t dgj = computationInRange ? dg_m[stage * t + k + i] : static_cast<data_t>(0);
            const data_t invj = computationInRange ? inv_m[stage * t + k + i] : static_cast<data_t>(0);

            QT[i] += dfi * dgj + dfj * dgi;
            P[i] = QT[i] * invi * invj;

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

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    #pragma HLS INLINE
    return sqrt(static_cast<data_t>(2 * m * (1 - PearsonCorrelation)));
}

void StreamToMemoryElement(stream<data_t, stream_d> &QT, stream<data_t, stream_d> &df, stream<data_t, stream_d> &dg, stream<data_t, stream_d> &inv, 
                           stream<aggregate_t, stream_d> &scatterLane, stream<aggregate_t, stream_d> &reductionLane, data_t *MP, index_t *MPI) {
    StreamToMemorySink:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1

        QT.read();
        df.read(); dg.read(); inv.read();
        scatterLane.read();
    }

    StreamToMemoryReduce:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1

        aggregate_t aggregate = reductionLane.read();
        MP[k] = PearsonCorrelationToEuclideanDistance(aggregate.value);
        MPI[k] = aggregate.index;
    }
}

void MatrixProfileKernelTLF(const data_t *QT, const data_t *df, const data_t *dg,
                            const data_t *inv, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem2

    #pragma HLS DATAFLOW

    constexpr index_t numStages = (n - m + 1 + t - 1) / t;

    // Streams required to calculate Correlations
    stream<data_t, stream_d> QT_s[numStages + 1];
    stream<data_t, stream_d> df_s[numStages + 1], dg_s[numStages + 1], inv_s[numStages + 1];

    // Store the intermediate results
    stream<aggregate_t, stream_d> scatterLane[numStages + 1], reductionLane[numStages + 1];

    MemoryToStreamElement(QT, df, dg, inv, QT_s[0], df_s[0], dg_s[0], inv_s[0], scatterLane[0], reductionLane[0]);

    for (index_t stage = 0; stage < numStages; ++stage) {
        #pragma HLS UNROLL
        DiagonalComputeElement(stage, QT_s[stage], df_s[stage], dg_s[stage], inv_s[stage], scatterLane[stage], reductionLane[stage], 
                               QT_s[stage + 1], df_s[stage + 1], dg_s[stage + 1], inv_s[stage + 1], scatterLane[stage + 1], reductionLane[stage + 1]);
    }

    StreamToMemoryElement(QT_s[numStages], df_s[numStages], dg_s[numStages], inv_s[numStages], scatterLane[numStages], reductionLane[numStages], MP, MPI);
}