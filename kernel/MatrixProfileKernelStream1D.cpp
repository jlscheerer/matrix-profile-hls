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

void MemoryToStreamElement(const data_t *T, stream<data_t, stream_d> &QT, stream<data_t, stream_d> &df_s, stream<data_t, stream_d> &dg_s,
                    stream<data_t, stream_d> &inv_s, stream<aggregate_t, stream_d> &scatterLane, stream<aggregate_t, stream_d> &reductionLane) {
    // store the previous (m-1) T-values in local "cache" (acts as shift-register)
    data_t T_m[m];
    #pragma HLS ARRAY_PARITION variable=T_m complete

    // store the first m T values in local "cache" (required for convolution)
    data_t Ti_m[m];
    #pragma HLS ARRAY_PARITION variable=Ti_m complete

    data_t inv_sum = 0;
    data_t qt_sum = 0;

    PrecomputationInitT:
    for (index_t i = 0; i < m; ++i) {
        #pragma HLS PIPELINE II=1
        data_t T_i = T[i];
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }

    data_t mean = 0;
    PrecomputationInitMu:
    for (index_t i = 0; i < m; ++i) {
        #pragma HLS UNROLL
        mean += T_m[i];
    }
    mean /= m;
    const data_t mu0 = mean;

    PrecomputationInitInvQT:
    for (index_t k = 0; k < m; ++k) {
        #pragma HLS UNROLL
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }
    data_t inv = static_cast<data_t>(1) / static_cast<data_t>(sqrt(inv_sum));
    const data_t inv0 = inv;

    QT.write(qt_sum);
    df_s.write(0); dg_s.write(0); inv_s.write(inv);

    // Will always be 1 & contained in the exclusionZone
    scatterLane.write(aggregate_t_init);

    PrecomputationCompute:
    for (index_t i = m; i < n; ++i) {
        data_t Ti = T[i];
        data_t Tm = T_m[0];

        mean = 0;
        PrecomputationComputeUpdateMean:
        for(index_t k = 1; k < m; ++k) {
            mean += T_m[k];
        }
        data_t prev_mean = mean;
        prev_mean += Tm; prev_mean /= m;
        mean += Ti; mean /= m;

        inv_sum = 0; qt_sum = 0;
        PrecomputationComputeUpdateInvQT:
        for (index_t k = 1; k < m; ++k) {
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }
        qt_sum += (Ti - mean) * (Ti_m[m - 1] - mu0);
        inv_sum += (Ti - mean) * (Ti - mean);
        inv = static_cast<data_t>(1) / sqrt(inv_sum);

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        data_t df = (Ti - Tm) / 2;

        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        data_t dg = (Ti - mean) + (Tm - prev_mean);

        QT.write(qt_sum);
        df_s.write(df); dg_s.write(dg); inv_s.write(inv);

        bool exclusionZone = (i - m + 1 < m/4);
        scatterLane.write(!exclusionZone ? (aggregate_t){qt_sum * inv * inv0, 0} : aggregate_t_init);

        // shift all values in T_m back
        PrecomputationComputeShift:
        for (index_t k = 0; k < m - 1; ++k) {
            #pragma HLS UNROLL
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = Ti;
    }

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

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem2

    #pragma HLS DATAFLOW

    constexpr index_t numStages = (n - m + 1 + t - 1) / t;

    // Streams required to calculate Correlations
    stream<data_t, stream_d> QT[numStages + 1];
    stream<data_t, stream_d> df[numStages + 1], dg[numStages + 1], inv[numStages + 1];

    // Store the intermediate results
    stream<aggregate_t, stream_d> scatterLane[numStages + 1], reductionLane[numStages + 1];

    MemoryToStreamElement(T, QT[0], df[0], dg[0], inv[0], scatterLane[0], reductionLane[0]);

    for (index_t stage = 0; stage < numStages; ++stage) {
        #pragma HLS UNROLL
        DiagonalComputeElement(stage, QT[stage], df[stage], dg[stage], inv[stage], scatterLane[stage], reductionLane[stage], QT[stage + 1],
                               df[stage + 1], dg[stage + 1], inv[stage + 1], scatterLane[stage + 1], reductionLane[stage + 1]);
    }

    StreamToMemoryElement(QT[numStages], df[numStages], dg[numStages], inv[numStages], scatterLane[numStages], reductionLane[numStages], MP, MPI);
}
