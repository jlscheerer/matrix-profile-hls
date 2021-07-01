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

void MemoryToStreamElement(const data_t *T, stream<data_t, stream_d> &QT, stream<data_t, stream_d> &df_i, stream<data_t, stream_d> &df_j,
                    stream<data_t, stream_d> &dg_i, stream<data_t, stream_d> &dg_j, stream<data_t, stream_d> &inv_i,
                    stream<data_t, stream_d> &inv_j, stream<aggregate_t, stream_d> &rowAggregate, stream<aggregate_t, stream_d> &columnAggregate) {
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
    data_t mu0 = mean;

    PrecomputationInitInvQT:
    for (index_t k = 0; k < m; ++k) {
        #pragma HLS UNROLL
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }
    data_t inv = 1 / sqrt(inv_sum);

    QT.write(qt_sum);
    df_i.write(0); dg_i.write(0); inv_i.write(inv);
    df_j.write(0); dg_j.write(0); inv_j.write(inv);

    PrecomputationCompute:
    for (index_t i = m; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        data_t Ti = T[i];
        data_t Tm = T_m[0];

        // recompute mean to achieve II=1
        mean = 0;
        PrecomputationComputeUpdateMean:
        for(index_t k = 1; k < m; ++k) {
            #pragma HLS UNROLL
            mean += T_m[k];
        }
        data_t prev_mean = mean;
        prev_mean += Tm; prev_mean /= m;
        mean += Ti; mean /= m;

        inv_sum = 0; qt_sum = 0;
        PrecomputationComputeUpdateInvQT:
        for (index_t k = 1; k < m; ++k) {
            #pragma HLS UNROLL
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }
        qt_sum += (Ti - mean) * (Ti_m[m - 1] - mu0);
        inv_sum += (Ti - mean) * (Ti - mean);
        inv = 1 / sqrt(inv_sum);

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        data_t df = (Ti - Tm) / 2;

        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        data_t dg = (Ti - mean) + (Tm - prev_mean);

        QT.write(qt_sum);
        df_i.write(df); dg_i.write(dg); inv_i.write(inv);
        df_j.write(df); dg_j.write(dg); inv_j.write(inv);

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
        
        columnAggregate.write(aggregate_t_init);
        rowAggregate.write(aggregate_t_init);
    }
    // =============== [/Reduce] ===============
}

// TODO: Potentially Template this function to avoid unnecessary memory/ressource consumption of "caches"
void DiagonalComputeElement(const index_t stage, stream<data_t, stream_d> &QT_in, stream<data_t, stream_d> &df_i_in, stream<data_t, stream_d> &df_j_in, 
                                     stream<data_t, stream_d> &dg_i_in, stream<data_t, stream_d> &dg_j_in, stream<data_t, stream_d> &inv_i_in, stream<data_t, stream_d> &inv_j_in, 
                                     stream<aggregate_t, stream_d> &rowAggregate_in, stream<aggregate_t, stream_d> &columnAggregate_in, stream<data_t, stream_d> &QT_out,
                                     stream<data_t, stream_d> &df_i_out, stream<data_t, stream_d> &df_j_out, stream<data_t, stream_d> &dg_i_out, stream<data_t, stream_d> &dg_j_out,
                                     stream<data_t, stream_d> &inv_i_out, stream<data_t, stream_d> &inv_j_out, stream<aggregate_t, stream_d> &rowAggregate_out, stream<aggregate_t, stream_d> &columnAggregate_out) {
    // TODO: Rename df_i_in to dfi_in to be more consistent with Stream2D

    // local "cache" [size: (n - m + 1 - stage) would be sufficient]
    data_t dfi_m[n - m + 1], dgi_m[n - m + 1], invi_m[n - m + 1];
    data_t dfj_m[n - m + 1], dgj_m[n - m + 1], invj_m[n - m + 1];

    aggregate_t columnAggregate_m[n - m + 1], rowAggregate_m[n - m + 1];

    data_t QT = 0, PearsonCorrelation = 0;

    // =============== [Scatter] ===============
    MatrixProfileScatter:
    for (index_t k = 0; k < n - m + 1 - stage; ++k) {
        #pragma HLS PIPELINE II=1

        data_t QTforward = QT_in.read();
        data_t dfi = df_i_in.read(), dgi = dg_i_in.read(), invi = inv_i_in.read();
        data_t dfj = df_j_in.read(), dgj = dg_j_in.read(), invj = inv_j_in.read();

        // store values in local "cache"
        QT = (k == 0) ? QTforward : QT;
        dfi_m[k] = dfi; dgi_m[k] = dgi; invi_m[k] = invi;
        dfj_m[k] = dfj; dgj_m[k] = dgj; invj_m[k] = invj;

        // forward values to subsequent processing elements
        if (k != 0)
            QT_out.write(QTforward);

        if (k != n - m - stage) {
            // forward everything but the last element for the row
            df_i_out.write(dfi); dg_i_out.write(dgi); inv_i_out.write(invi);
        }

        if (k != 0) {
            // forward everything but the first element for the columns
            df_j_out.write(dfj); dg_j_out.write(dgj); inv_j_out.write(invj);
        }
    }
    // =============== [/Scatter] ===============

    // =============== [Compute] ===============
    // check if processing element is within the exclusion zone
    // Exclusion Zone <==> i - m/4 <= j <= i + m/4
    // 				  <==> j <= i + m/4 [i <= j, m > 0]
    // 				  <==> (j - i) <= m / 4
    //				  <==> stage <= m/4
    const bool exclusionZone = stage < m/4;
    MatrixProfileCompute:
    for (index_t k = 0; k < n - m + 1 - stage; ++k) {
        #pragma HLS PIPELINE II=1

        data_t dfi = dfi_m[k], dgi = dgi_m[k], invi = invi_m[k];
        data_t dfj = dfj_m[k], dgj = dgj_m[k], invj = invj_m[k];

        QT += dfi * dgj + dfj * dgi;
        PearsonCorrelation = QT * invi * invj;

        rowAggregate_m[k] = !exclusionZone ? (aggregate_t) {PearsonCorrelation, stage + k} : aggregate_t_init;
        columnAggregate_m[k] = !exclusionZone ? (aggregate_t) {PearsonCorrelation, k} : aggregate_t_init;
    }
    // =============== [/Compute] ===============

    // =============== [Reduce] ===============
    MatrixProfileReduce:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1
        // get previous aggregates from predecessor
        aggregate_t prevRowAggregate = rowAggregate_in.read();
        aggregate_t prevColumnAggregate = columnAggregate_in.read();
        // get aggregates computed in the current processing element (if applicable)
        aggregate_t rowAggregate = (k <= n - m - stage) ? rowAggregate_m[k] : aggregate_t_init;
        aggregate_t columnAggregate = (k >= stage) ? columnAggregate_m[k - stage] : aggregate_t_init;
        // forward reduced aggregates
        rowAggregate_out.write(rowAggregate.value > prevRowAggregate.value ? rowAggregate : prevRowAggregate);
        columnAggregate_out.write(columnAggregate.value > prevColumnAggregate.value ? columnAggregate : prevColumnAggregate);
    }
    // =============== [/Reduce] ===============
}

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    #pragma HLS INLINE
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void StreamToMemoryElement(stream<aggregate_t, stream_d> &rowAggregates, stream<aggregate_t, stream_d> &columnAggregates, data_t *MP, index_t *MPI) {
    // TODO: Add Comment(s)
    StreamToMemoryReduce:
    for (index_t k = 0; k < n - m + 1; ++k) {
        #pragma HLS PIPELINE II=1

        aggregate_t rowAggregate = rowAggregates.read();
        aggregate_t columnAggregate = columnAggregates.read();

        MP[k] = PearsonCorrelationToEuclideanDistance((rowAggregate.value > columnAggregate.value) ? rowAggregate.value : columnAggregate.value);
        MPI[k] = (rowAggregate.value > columnAggregate.value) ? rowAggregate.index : columnAggregate.index;
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem2

    #pragma HLS DATAFLOW

    constexpr index_t numStages = n - m + 1;

    // Streams required to calculate Correlations
    stream<data_t, stream_d> QT[numStages + 1];
    stream<data_t, stream_d> df_i[numStages + 1], dg_i[numStages + 1], inv_i[numStages + 1];
    stream<data_t, stream_d> df_j[numStages + 1], dg_j[numStages + 1], inv_j[numStages + 1];

    // Store the intermediate results
    stream<aggregate_t, stream_d> rowAggregate[numStages + 1], columnAggregate[numStages + 1];

    MemoryToStreamElement(T, QT[0], df_i[0], df_j[0], dg_i[0], dg_j[0], inv_i[0], inv_j[0], rowAggregate[0], columnAggregate[0]);

    for (index_t stage = 0; stage < numStages; ++stage) {
        #pragma HLS UNROLL
        DiagonalComputeElement(stage, QT[stage], df_i[stage], df_j[stage], dg_i[stage], dg_j[stage],
                                        inv_i[stage], inv_j[stage], rowAggregate[stage], columnAggregate[stage], QT[stage + 1],
                                        df_i[stage + 1], df_j[stage + 1], dg_i[stage + 1], dg_j[stage + 1], inv_i[stage + 1],
                                        inv_j[stage + 1], rowAggregate[stage + 1], columnAggregate[stage + 1]);
    }

    StreamToMemoryElement(rowAggregate[numStages], columnAggregate[numStages], MP, MPI);
}
