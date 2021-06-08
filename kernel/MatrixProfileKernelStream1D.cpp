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

static constexpr size_t stream_d = 2;

void MemoryToStream(const data_t *T, stream<data_t, stream_d> &QT, stream<data_t, stream_d> &df_i, stream<data_t, stream_d> &df_j,
                    stream<data_t, stream_d> &dg_i, stream<data_t, stream_d> &dg_j, stream<data_t, stream_d> &inv_i,
                    stream<data_t, stream_d> &inv_j, stream<aggregate_t, stream_d> &rowAggregate, stream<aggregate_t, stream_d> &columnAggregate) {
    data_t mean = 0;
    data_t inv_sum = 0;
    data_t qt_sum = 0;

    // use T_m as shift register containing the previous m T elements
    // need to be able to access these elements with no contention
    data_t T_m[m];

    // the first m T values, required for convolution
    data_t Ti_m[m];

    // move out mean pipeline & unroll
    for (index_t i = 0; i < m; i++) {
        data_t T_i = T[i];
        mean += T_i;
        T_m[i] = T_i;
        Ti_m[i] = T_i;
    }
    mean /= m;

    // Prepare for the first values (QT, inv, Aggregates)
    // Calculate initial values
    data_t mu0 = mean;

    // TODO: Unroll this Loop
    for (index_t k = 0; k < m; k++) {
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        qt_sum += (T_m[k] - mean) * (Ti_m[k] - mu0);
    }
    data_t inv = 1 / sqrt(inv_sum);

    QT.write(qt_sum);

    inv_i.write(inv);
    inv_j.write(inv);

    columnAggregate.write(aggregate_t_init);
    rowAggregate.write(aggregate_t_init);

    data_t prev_mean = 0;
    for (index_t i = m; i < n; ++i) {
        data_t T_i = T[i];
        data_t T_r = T_m[0];

        // TODO: recompute mean to achieve II=1
        // set and update mean
        prev_mean = mean;
        mean = mean + (T_i - T_r) / m;
        // mu[i - m + 1] = mean;

        inv_sum = 0;
        qt_sum = 0;
        // TODO: needs to be unrolled
        for (index_t k = 1; k < m; k++) {
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
            qt_sum += (T_m[k] - mean) * (Ti_m[k - 1] - mu0);
        }
        // perform last element of the loop separately (this requires the newly read value)
        qt_sum += (T_i - mean) * (Ti_m[m - 1] - mu0);

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        // df[i - m + 1] = (T_i - T_r) / 2;
        data_t df = (T_i - T_r) / 2;

        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        // dg[i - m + 1] = (T_i - mean) + (T_r - prev_mean);
        data_t dg = (T_i - mean) + (T_r - prev_mean);

        inv_sum += (T_i - mean) * (T_i - mean);
        inv = 1 / sqrt(inv_sum);

        QT.write(qt_sum);

        df_i.write(df);
        dg_i.write(dg);
        inv_i.write(inv);

        df_j.write(df);
        dg_j.write(dg);
        inv_j.write(inv);

        columnAggregate.write(aggregate_t_init);
        rowAggregate.write(aggregate_t_init);

        // shift all values in T_m back
        // TODO: Unroll
        for (index_t k = 0; k < m - 1; k++) {
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = T_i;
    }
}

void MatrixProfileComputationElement(index_t stage, stream<data_t, stream_d> &QT_in, stream<data_t, stream_d> &df_i_in, stream<data_t, stream_d> &df_j_in, 
                                     stream<data_t, stream_d> &dg_i_in, stream<data_t, stream_d> &dg_j_in, stream<data_t, stream_d> &inv_i_in, stream<data_t, stream_d> &inv_j_in, 
                                     stream<aggregate_t, stream_d> &rowAggregate_in, stream<aggregate_t, stream_d> &columnAggregate_in, stream<data_t, stream_d> &QT_out,
                                     stream<data_t, stream_d> &df_i_out, stream<data_t, stream_d> &df_j_out, stream<data_t, stream_d> &dg_i_out, stream<data_t, stream_d> &dg_j_out,
                                     stream<data_t, stream_d> &inv_i_out, stream<data_t, stream_d> &inv_j_out, stream<aggregate_t, stream_d> &rowAggregate_out, stream<aggregate_t, stream_d> &columnAggregate_out) {
    for (index_t i = 0; i < stage; ++i) {
        // forward column aggregate
        columnAggregate_out.write(columnAggregate_in.read());
    }

    // take care of the first iteration
    data_t QT = QT_in.read();
    data_t inv_i = inv_i_in.read();
    data_t inv_j = inv_j_in.read();

    // forward the inv value (row) (if we are not the last element)
    if (stage != n - m)
        inv_i_out.write(inv_i);

    aggregate_t columnAggregate = columnAggregate_in.read();
    aggregate_t rowAggregate = rowAggregate_in.read();

    data_t PearsonCorrelation = QT * inv_i * inv_j;
    
    // Check If ProcessingElement is within the ExclusionZone
    // Exclusion Zone <==> i - m/4 <= j <= i + m/4
    // 				  <==> j <= i + m/4 [i <= j, m > 0]
    // 				  <==> (j - i) <= m / 4
    //				  <==> stage <= m/4
    bool exclusionZone = stage < m/4;

    // pass on the column aggregate
    if (!exclusionZone && PearsonCorrelation >= columnAggregate.value) {
        // use our value
        // remember "best" row for columns
        columnAggregate_out.write({PearsonCorrelation, 0});
    } else {
        // pass on previous aggregate
        columnAggregate_out.write(columnAggregate);
    }

    // pass on the row aggregate
    if (!exclusionZone && PearsonCorrelation >= rowAggregate.value) {
        // use our value
        rowAggregate_out.write({PearsonCorrelation, stage});
    } else {
        // pass on previous aggregate
        rowAggregate_out.write(rowAggregate);
    }

    // n - m + 1 - stage - 1 because first element was taken care outside the loop
    for (index_t i = 1; i < n - m + 1 - stage; ++i) {
        data_t QT_forward = QT_in.read();

        // read values concerning the current row
        data_t df_i = df_i_in.read();
        data_t dg_i = dg_i_in.read();
        inv_i = inv_i_in.read();

        // read values concerining the current column
        data_t df_j = df_j_in.read();
        data_t dg_j = dg_j_in.read();
        inv_j = inv_j_in.read();

        rowAggregate = rowAggregate_in.read();
        columnAggregate = columnAggregate_in.read();

        // forward initial values for QT
        QT_out.write(QT_forward);

        // Calculate new values of QT: QT_{i, j} = QT_{i-1, j-1} - df_{i} * dg_{j} + df_{j} * dg_{i}
        QT = QT + df_i * dg_j + df_j * dg_i;

        // Calculate Pearson Correlation: QT * (1 / (||T_{i, m} - μ[i]||)) * (1 / (||T_{j, m} - μ[j]||))
        PearsonCorrelation = QT * inv_i * inv_j;

        // pass on the column aggregate
        if (!exclusionZone && PearsonCorrelation >= columnAggregate.value) {
            // use our value
            // remember "best" row for columns
            columnAggregate_out.write({PearsonCorrelation, i});
        } else {
            // pass on previous aggregate
            columnAggregate_out.write(columnAggregate);
        }

        // pass on the row aggregate
        if (!exclusionZone && PearsonCorrelation >= rowAggregate.value) {
            // use our value
            // remember "best" column for rows
            rowAggregate_out.write({PearsonCorrelation, i + stage});
        } else {
            // pass on previous aggregate
            rowAggregate_out.write(rowAggregate);
        }

        // Move elements along
        if (i != (n - m + 1) - 1 - stage) {
            // pass on everything but the last element
            df_i_out.write(df_i);
            dg_i_out.write(dg_i);
            inv_i_out.write(inv_i);
        }

        inv_j_out.write(inv_j);

        if (i != 1) {
            // pass on everything but the first element
            df_j_out.write(df_j);
            dg_j_out.write(dg_j);
        }

    }

    for (index_t i = 0; i < stage; ++i) {
        // forward row aggregate
        rowAggregate_out.write(rowAggregate_in.read());
    }
}

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void StreamToMemory(stream<aggregate_t, stream_d> &rowAggregate, stream<aggregate_t, stream_d> &columnAggregate, data_t *MP, index_t *MPI) {
     // local "cache" storing the column aggregates (to merge them later)
    aggregate_t aggregates_m[n - m + 1];

    // read column-wise aggregates and cache
    for (index_t i = 0; i < n - m + 1; ++i) {
        aggregates_m[i] = columnAggregate.read();
    }

    // read row-wise aggreagtes and merge
    for (index_t i = 0; i < n - m + 1; ++i) {
        aggregate_t rAggregate = rowAggregate.read();
        aggregate_t cAggregate = aggregates_m[i];
        MP[i] = PearsonCorrelationToEuclideanDistance(rAggregate.value > cAggregate.value ? rAggregate.value : cAggregate.value);
        MPI[i] = rAggregate.value > cAggregate.value ? rAggregate.index : cAggregate.index;
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    constexpr index_t numStages = n - m + 1;

    // Streams required to calculate Correlations
    stream<data_t, stream_d> QT[numStages + 1];

    stream<data_t, stream_d> df_i[numStages + 1];
    stream<data_t, stream_d> dg_i[numStages + 1];
    stream<data_t, stream_d> inv_i[numStages + 1];

    stream<data_t, stream_d> df_j[numStages + 1];
    stream<data_t, stream_d> dg_j[numStages + 1];
    stream<data_t, stream_d> inv_j[numStages + 1];

    // Store the intermediate results
    stream<aggregate_t, stream_d> rowAggregate[numStages + 1];
    stream<aggregate_t, stream_d> columnAggregate[numStages + 1];

    MemoryToStream(T, QT[0], df_i[0], df_j[0], dg_i[0], dg_j[0], inv_i[0], inv_j[0], rowAggregate[0], columnAggregate[0]);

    for (index_t stage = 0; stage < numStages; ++stage) {
        MatrixProfileComputationElement(stage, QT[stage], df_i[stage], df_j[stage], dg_i[stage], dg_j[stage],
                                        inv_i[stage], inv_j[stage], rowAggregate[stage], columnAggregate[stage], QT[stage + 1],
                                        df_i[stage + 1], df_j[stage + 1], dg_i[stage + 1], dg_j[stage + 1], inv_i[stage + 1],
                                        inv_j[stage + 1], rowAggregate[stage + 1], columnAggregate[stage + 1]);
    }

    StreamToMemory(rowAggregate[numStages], columnAggregate[numStages], MP, MPI);
}