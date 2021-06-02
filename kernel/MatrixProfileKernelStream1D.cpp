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

static constexpr size_t stream_d = 8;

typedef struct {
    data_t df, dg, inv;
} compute_t;

void MemoryToStream(const data_t *T, stream<data_t, stream_d> &QT, stream<compute_t, stream_d> &compute_i, stream<compute_t, stream_d> &compute_j,
        stream<aggregate_t, stream_d> &rowAggregate, stream<aggregate_t, stream_d> &columnAggregate) {
    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];

    // the first m T values, required for convolution
    data_t Ti_m[m];

    // TODO: Move mean out to unroll
    data_t mean = 0;
    for (index_t i = 0; i < m; i++) {
        data_t T_i = T[i];
        mean += T_i;
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }
    mean /= m;
    data_t mu0 = mean;

    // TODO: Unroll this Loop
    data_t inv_sum = 0, qt_sum = 0;
    for (index_t k = 0; k < m; k++) {
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }
    data_t inv = 1 / sqrt(inv_sum);

    QT.write(qt_sum);

    compute_i.write({0, 0, inv}); // df_i, dg_i, inv_i
    compute_j.write({0, 0, inv}); // df_j, dg_j, inv_j

    rowAggregate.write({aggregate_init, index_init});
    columnAggregate.write({aggregate_init, index_init});

    data_t prev_mean;
    for (index_t i = m; i < n; ++i) {
        data_t T_i = T[i];
        data_t T_r = T_m[0];

        // set and update mean
        // TODO: Recompute Mean to achieve II=1
        prev_mean = mean;
        // mu[i - m + 1] = mean;
        mean = mean + (T_i - T_r) / m;

        inv_sum = 0; qt_sum = 0;
        // TODO: needs to be unrolled
        for (index_t k = 1; k < m; k++) {
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }
        // perform last element of the loop separately (this requires the newly read value)
        qt_sum += (T_i - mean) * (Ti_m[m - 1] - mu0);
        // QT[i - m + 1] = qt_sum;
        QT.write(qt_sum);

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        // df[i - m + 1] = (T_i - T_r) / 2;
        data_t df = (T_i - T_r) / 2;
        
        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        // dg[i - m + 1] = (T_i - mean) + (T_r - prev_mean);
        data_t dg = (T_i - mean) + (T_r - prev_mean);
        
        inv_sum += (T_i - mean) * (T_i - mean);
        inv = 1 / sqrt(inv_sum);

        compute_i.write({df, dg, inv}); // df_i, dg_i, inv_i
        compute_j.write({df, dg, inv}); // df_j, dg_j, inv_j

        rowAggregate.write({aggregate_init, index_init});
        columnAggregate.write({aggregate_init, index_init});

        // shift all values in T_m back
        // TODO: Unroll
        for (index_t k = 0; k < m - 1; k++) {
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = T_i;
    }
}

const inline bool ExclusionZone(const index_t i, const index_t j) {
    // Exclusion Zone <==> i - m/4 <= j <= i + m/4
    // 				  <==> j <= i + m/4 [i <= j, m > 0]
    return j < i + m / 4;
}

// TODO: Can optimize exclusionZone either every result will be in the exclusionZone or none
void MatrixProfileComputationElement(index_t stage, stream<data_t, stream_d> &QT_in,
        stream<compute_t, stream_d> &compute_i_in, stream<compute_t, stream_d> &compute_j_in,
        stream<aggregate_t, stream_d> &rowAggregate_in, stream<aggregate_t, stream_d> &columnAggregate_in,
        stream<data_t, stream_d> &QT_out, stream<compute_t, stream_d> &compute_i_out, stream<compute_t, stream_d> &compute_j_out,
        stream<aggregate_t, stream_d> &rowAggregate_out, stream<aggregate_t, stream_d> &columnAggregate_out) {
    for (index_t i = 0; i < stage; ++i) {
        // forward column aggregate
        aggregate_t columnAggregate = columnAggregate_in.read();
        columnAggregate_out.write(columnAggregate);
    }

    // take care of the first iteration
    data_t QT = QT_in.read();
    compute_t compute_i = compute_i_in.read();
    compute_t compute_j = compute_j_in.read();

    aggregate_t rowAggregate = rowAggregate_in.read();
    aggregate_t columnAggregate = columnAggregate_in.read();
    data_t P = QT * compute_i.inv * compute_j.inv;
    
    bool exclusionZone = ExclusionZone(0, stage);

    // pass on the column aggregate (first column)
    if (!exclusionZone && P >= columnAggregate.value) {
        // use our value
        columnAggregate_out.write({P, 0}); // remember "best" row for columns
    } else {
        // pass on previous aggregate
        columnAggregate_out.write(columnAggregate);
    }

    // previous row values hold for one iteration of the loop
    compute_t pcompute_i = compute_i;
    data_t pP = P; bool pExclusionZone = exclusionZone;
    aggregate_t pRowAggregate = rowAggregate;
    for (index_t i = 1; i < n - m + 1 - stage; ++i) {
        data_t pQT = QT_in.read();

        compute_i = compute_i_in.read();
        compute_j = compute_j_in.read();

        rowAggregate = rowAggregate_in.read();
        columnAggregate = columnAggregate_in.read();

        QT_out.write(pQT);
        compute_i_out.write(pcompute_i);
        compute_j_out.write(compute_j);

        QT = QT + compute_i.df * compute_j.dg + compute_j.df * compute_i.dg;
        P = QT * compute_i.inv * compute_j.inv;
        exclusionZone = ExclusionZone(i, stage + i);

        // pass on the column aggregate
        if (!exclusionZone && P >= columnAggregate.value) {
            // use our value
            columnAggregate_out.write({P, static_cast<index_t>(i)}); // remember "best" row for columns
        } else {
            // pass on previous aggregate
            columnAggregate_out.write(columnAggregate);
        }

        // pass on the row aggregate
        if (!pExclusionZone && pP >= pRowAggregate.value) {
            // use our value
            rowAggregate_out.write({pP, static_cast<index_t>(i + stage - 1)}); // remember "best" column // for rows
        } else {
            // pass on previous aggregate
            rowAggregate_out.write(pRowAggregate);
        }

        // Shift everything backward
        pcompute_i = compute_i;
        pP = P; pExclusionZone = exclusionZone;
        pRowAggregate = rowAggregate;
    }

    // pass on the row aggregate (last row)
    if (!exclusionZone && pP >= pRowAggregate.value) {
        // use our value
        rowAggregate_out.write({pP, n - m}); // remember "best" column for rows
    } else {
        // pass on previous aggregate
        rowAggregate_out.write(pRowAggregate);
    }

    // forward row aggregates
    for (index_t i = 0; i < stage; ++i) {
        rowAggregate = rowAggregate_in.read();
        rowAggregate_out.write(rowAggregate);
    }
}

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void StreamToMemory(stream<aggregate_t, stream_d> &rowAggregate, stream<aggregate_t, stream_d> &columnAggregate, data_t *MP, index_t *MPI) {
    // TODO: Use local cache don't directly write to memory (might be better?) (this way we need to access 3 times!)
    for (index_t i = 0; i < n - m + 1; ++i){
        // TODO: If this remains move to constant
        MP[i] = 1e12; // i.e. "positive infinity"
        MPI[i] = -1;
    }

    // Just columns
    for (index_t i = 0; i < n - m + 1; ++i) {
        aggregate_t aggregate = columnAggregate.read();
        // TODO: Improve this (move out MP?, calculate PearsonCorrelation only once)
        data_t euclideanDistance = PearsonCorrelationToEuclideanDistance(aggregate.value);
        if (euclideanDistance < MP[i]) {
            MP[i] = euclideanDistance;
            MPI[i] = aggregate.index;
        }
    }

    // Just rows
    for (index_t i = 0; i < n - m + 1; ++i) {
        aggregate_t aggregate = rowAggregate.read();
        // TODO: Improve this (move out MP?, calculate PearsonCorrelation only once)
        data_t euclideanDistance = PearsonCorrelationToEuclideanDistance(aggregate.value);
        if (euclideanDistance < MP[i]) {
            MP[i] = euclideanDistance;
            MPI[i] = aggregate.index;
        }
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi     port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi     port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=MPI offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=T   bundle=control
    #pragma HLS INTERFACE s_axilite port=MP  bundle=control
    #pragma HLS INTERFACE s_axilite port=MPI bundle=control
    
    #pragma HLS DATAFLOW

    constexpr index_t numStages = n - m + 1;

    // Streams for the Scatter Lane (i: rows, j: columns)
    stream<data_t, stream_d> QT[numStages + 1];
    stream<compute_t, stream_d> compute_i[numStages + 1];
    stream<compute_t, stream_d> compute_j[numStages + 1];

    // Streams for the Reduction Lane (Aggregates)
    stream<aggregate_t, stream_d> rowAggregate[numStages + 1];
    stream<aggregate_t, stream_d> columnAggregate[numStages + 1];

    MemoryToStream(T, QT[0], compute_i[0], compute_j[0], rowAggregate[0], columnAggregate[0]);

    for (index_t stage = 0; stage < numStages; ++stage) {
        #pragma HLS UNROLL
        MatrixProfileComputationElement(stage, QT[stage], compute_i[stage], compute_j[stage], rowAggregate[stage], columnAggregate[stage], 
                QT[stage + 1], compute_i[stage + 1], compute_j[stage + 1], rowAggregate[stage + 1], columnAggregate[stage + 1]);
    }

    StreamToMemory(rowAggregate[numStages], columnAggregate[numStages], MP, MPI);
}
