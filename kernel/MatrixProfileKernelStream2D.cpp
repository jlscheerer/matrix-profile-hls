/**
 * @file    MatrixProfileKernelStream2D.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Stream-2D]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"

    #include "hls_math.h"
    #include "hls_stream.h"

    #include "kernel/HLSMathUtil.hpp"

    using hls::stream;
#endif

static constexpr size_t stream_d = 3;

// number of tiles in the first row
// ⌈(n - m + 1) / t⌉ = ⌈sublen / t⌉
static constexpr index_t nTiles = (sublen + t - 1) / t;

static constexpr index_t min(const index_t a, const index_t b){ return (a < b) ? a : b; }
static constexpr index_t max(const index_t a, const index_t b){ return (a > b) ? a : b; }

// https://math.stackexchange.com/questions/2134011/conversion-of-upper-triangle-linear-index-from-index-on-symmetrical-array
static constexpr index_t index2D(const index_t x, const index_t y) { return (nTiles * (nTiles - 1)) / 2 - ((nTiles - y) * (nTiles - y - 1)) / 2 + x; }

void MemoryToStreamElement(const data_t *T, stream<data_t, stream_d> &sT, stream<data_t, stream_d> &sMu,
                    stream<data_t, stream_d> &sDf, stream<data_t, stream_d> &sDg, stream<data_t, stream_d> &sInv) {
    // store the previous (m-1) T-values in local "cache" (acts as shift-register)
    data_t T_m[m];
    #pragma HLS ARRAY_PARITION variable=T_m complete

    // initialize mean calculation and local "cache"
    // read T values cannot be directly passed on (this would cause a deadlock in the ProcessingElements)
    PrecomputationInitT:
    for (index_t i = 0; i < m; ++i) {
        #pragma HLS PIPELINE II=1
        T_m[i] = T[i];
    }

    data_t mean = 0;
    PrecomputationInitMu:
    for (index_t i = 0; i < m; ++i){
        #pragma HLS UNROLL
        mean += T_m[i];
    }
    mean /= m;

    data_t inv_sum = 0;
    PrecomputationInitInv:
    for (index_t k = 0; k < m; ++k){
        #pragma HLS UNROLL
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
    }

    // do the first iteration manually (mean, inv, df, dg)
    sMu.write(mean); sDf.write(0); sDg.write(0);
    sInv.write(static_cast<data_t>(1) / sqrt(inv_sum));

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

        sT.write(Tm); sMu.write(mean);

        // calculate df: (T[i+m-1] - T[i-1]) / 2
        data_t df = (Ti - Tm) / 2;
        sDf.write(df);

        // calculate dg: (T[i+m-1] - μ[i]) * (T[i-1] - μ[i-1])
        data_t dg = (Ti - mean) + (Tm - prev_mean);
        sDg.write(dg);

        inv_sum = 0;
        PrecomputationComputeUpdateInv:
        for (index_t k = 1; k < m; ++k){
            #pragma HLS UNROLL
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        }
        // perform last element of the loop separately (this requires the new val
        inv_sum += (Ti - mean) * (Ti - mean);
        sInv.write(static_cast<data_t>(1) / sqrt(inv_sum));

        // shift all values in T_m back
        PrecomputationComputeShift:
        for (index_t k = 0; k < m - 1; ++k){
            #pragma HLS UNROLL
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = Ti;
    }

    // Push the remaining values
    PrecomputationScatterCacheT:
    for (index_t i = 0; i < m; ++i) {
        #pragma HLS PIPELINE II=1
        sT.write(T_m[i]);
    }
}

void PrimaryDiagonalComputeElement(index_t yStage, index_t xStage, stream<data_t, stream_d> &sT_in,
        stream<data_t, stream_d> &sMu_in, stream<data_t, stream_d> &sDf_in, stream<data_t, stream_d> &sDg_in, stream<data_t, stream_d> &sInv_in,

        stream<data_t, stream_d> &Ti_out, stream<data_t, stream_d> &Tj_out, stream<data_t, stream_d> &mui_out, stream<data_t, stream_d> &muj_out,
        stream<data_t, stream_d> &dfi_out, stream<data_t, stream_d> &dfj_out, stream<data_t, stream_d> &dgi_out, stream<data_t, stream_d> &dgj_out,
        stream<data_t, stream_d> &invi_out, stream<data_t, stream_d> &invj_out,

        stream<aggregate_t, stream_d> &rowAggregate_out, stream<aggregate_t, stream_d> &columnAggregate_out,

        stream<data_t, stream_d> &sT_out, stream<data_t, stream_d> &sMu_out, stream<data_t, stream_d> &sDf_out, stream<data_t, stream_d> &sDg_out,
        stream<data_t, stream_d> &sInv_out) {
    // local "cache" for time series values for the current row/column
    data_t Ti_m[m], Tj_m[t + m - 1];

    // local "cache" for means for the current row/column
    data_t mui_m = 0, muj_m[t];
    
    // local "cache" for df/dg for the current row/column
    data_t dfi_m[t], dfj_m[2 * t - 1], dgi_m[t], dgj_m[2 * t - 1];

    // local "cache" for inverses for the current row/column
    data_t invi_m[t], invj_m[2 * t - 1];

    // factor=3 required for fadd and fmul (update if data_t changes)
    aggregate_t rowAggregate[t], columnAggregate[2 * t - 1];
    // =============== [Scatter] ===============
    data_t mu = 0, df = 0, dg = 0, inv = 0;
    ScatterDiagonalLane:
    for (index_t i = 0; i < n - t * xStage; ++i) {
        #pragma HLS PIPELINE II=1
        data_t Ti = sT_in.read();

        if (i < (n - m + 1 - t * xStage)) {
            mu = sMu_in.read();
            df = sDf_in.read();
            dg = sDg_in.read();
            inv = sInv_in.read();
        }

        if (i < m)
            Ti_m[i] = Ti;

        if (i < t + m - 1)
            Tj_m[i] = Ti;

        if (i == 0)
            mui_m = mu;

        if (i < t) {
            muj_m[i] = mu;
            dfi_m[i] = df;
            dgi_m[i] = dg;
            invi_m[i] = inv;
            rowAggregate[i] = aggregate_t_init;
        }

        if (i < 2 * t - 1) {
            dfj_m[i] = df;
            dgj_m[i] = dg;
            invj_m[i] = inv;
            columnAggregate[i] = aggregate_t_init;
        }

        // check that we are not the last element
        // if not we can forward elements to the remaining elements
        if (xStage < nTiles - 1) {
            if (i < m)
                Ti_out.write(Ti);

            if (i == 0)
                mui_out.write(mu);

            if (i < t) {
                dfi_out.write(df);
                dgi_out.write(dg);
                invi_out.write(inv);
            }

            if (i >= t) {
                sT_out.write(Ti);
                Tj_out.write(Ti);

                if (i < (n - m + 1 - t * xStage)) {
                    sMu_out.write(mu);
                    muj_out.write(mu);

                    sDf_out.write(df);
                    dfj_out.write(df);

                    sDg_out.write(dg);
                    dgj_out.write(dg);

                    sInv_out.write(inv);
                    invj_out.write(inv);
                }
            }
        }
    }
    // =============== [/Scatter] ===============

    // =============== [Compute] =============== 
    
    data_t QT[t], P[t];
    aggregate_t rowAggregate_m = aggregate_t_init;
    
    // Compute the Matrix Profile (here)
    const index_t yOffset = yStage * t;
    const index_t xOffset = xStage * t;

    // Exclusion Zone <==> realX - m/4 <= realY <= realX + m/4
    // 				  <==> (xStage * t + r + i) - m/4 <= yStage * t + r <= (xStage * t + r + i) + m/4
    // 				  <==> xStage * t + i - m/4 <= yStage * t <= xStage * t + i + m/4
    // 				  <==> xStage * t + i - m/4 <= yStage * t [t > 0, xStage >= yStage, i > 0, m/4 > 0]
    //    			  <==> (xStage - yStage) * t + i - m/4 <= 0
    // 				  <==> (xStage - yStage) * t - m/4 <= -i
    //   			  <==> i <= (yStage - xStage) * t + m/4
    const index_t exclusionZone = max((yStage - xStage) * t + m / 4, 0);

    // compute the first row of the matrix
    MatrixProfileComputeInitQRow:
    for (index_t i = exclusionZone; i < min(t, n - m + 1 - t * xStage); ++i) {
        // compute convolution explicitly
        data_t sum = 0;
        MatrixProfileComputeInitQColumn:
        for (index_t j = 0; j < m; j++) {
            sum += (Ti_m[j] - mui_m) * (Tj_m[i + j] - muj_m[i]);
        }
        QT[i] = sum;
        data_t PearsonCorrelation = QT[i] * invi_m[0] * invj_m[i];

        // update aggregates in case of improvement
        if (PearsonCorrelation > rowAggregate_m.value)
            rowAggregate_m = {PearsonCorrelation, static_cast<index_t>(i + xOffset)}; // remember "best" column for rows
        if (PearsonCorrelation != aggregate_init)
            columnAggregate[i] = {PearsonCorrelation, static_cast<index_t>(0 + yOffset)}; // remember "best" row for columns
    }
    rowAggregate[0] = rowAggregate_m;    

    // Compute (t-1) rows using the simplification
    MatrixProfileComputeRow:
    for (index_t r = 1; r < t; ++r) {
        rowAggregate_m = aggregate_t_init;
        MatrixProfileComputeColumn:
        for (index_t i = exclusionZone; i < min(t, n - m - t * xStage - r + 1); ++i) {
            #pragma HLS PIPELINE II=1
            
            // QT_{i, j} = QT_{i-1, j-1} + df_i * dg_j + df_j * dg_i
            // QT[i] was the previous value (i.e. value diagonally above the current QT[i])
            QT[i] = QT[i] + dfi_m[r] * dgj_m[i + r] + dfj_m[i + r] * dgi_m[r];

            // calculate Pearson Correlation
            P[i] = QT[i] * invi_m[r] * invj_m[i + r];
            
            if (P[i] > rowAggregate_m.value)
                rowAggregate_m = {P[i], static_cast<index_t>(r + i + xOffset)}; // remember "best" column for rows
            
            const index_t column = r + i;
            if (P[i] > columnAggregate[column].value)
                columnAggregate[column] = {P[i], static_cast<index_t>(r + yOffset)}; // remember "best" row for columns
        }
        rowAggregate[r] = rowAggregate_m;
    }

    // =============== [/Compute] ===============

    // =============== [Reduce] ===============
    // Don't need to forward anything because we are the first element of the row
    // Just row
    index_t rowAggregateLength = min(t, n - m + 1 - t * yStage);
    ScatterReduceAggregateRow:
    for (index_t i = 0; i < rowAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        rowAggregate_out.write(rowAggregate[i]);
    }
    // Just column
    index_t columnAggregateLength = min(t + (t - 1), n - m + 1 - t * yStage);
    ScatterReduceAggregateColumn:
    for (index_t i = 0; i < columnAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        columnAggregate_out.write(columnAggregate[i]);
    }
    // =============== [/Reduce] ===============
}

void DiagonalComputeElement(index_t yStage, index_t xStage, stream<data_t, stream_d> &Ti_in,
        stream<data_t, stream_d> &Tj_in, stream<data_t, stream_d> &mui_in, stream<data_t, stream_d> &muj_in, stream<data_t, stream_d> &dfi_in,
        stream<data_t, stream_d> &dfj_in, stream<data_t, stream_d> &dgi_in, stream<data_t, stream_d> &dgj_in, stream<data_t, stream_d> &invi_in,
        stream<data_t, stream_d> &invj_in,

        stream<aggregate_t, stream_d> &rowAggregate_in, stream<aggregate_t, stream_d> &columnAggregate_in,

        stream<data_t, stream_d> &Ti_out, stream<data_t, stream_d> &Tj_out, stream<data_t, stream_d> &mui_out, stream<data_t, stream_d> &muj_out,
        stream<data_t, stream_d> &dfi_out, stream<data_t, stream_d> &dfj_out, stream<data_t, stream_d> &dgi_out, stream<data_t, stream_d> &dgj_out,
        stream<data_t, stream_d> &invi_out,stream<data_t, stream_d> &invj_out,

        stream<aggregate_t, stream_d> &rowAggregate_out, stream<aggregate_t, stream_d> &columnAggregate_out) {
    // local "cache" for time series values for the current row/column
    data_t Ti_m[m], Tj_m[t + m - 1];

    // local "cache" for means for the current row/column
    data_t mui_m = 0, muj_m[t];
    
    // local "cache" for df/dg for the current row/column
    data_t dfi_m[t], dfj_m[2 * t - 1], dgi_m[t], dgj_m[2 * t - 1];

    // local "cache" for inverses for the current row/column
    data_t invi_m[t], invj_m[2 * t - 1];

    // factor=3 required for fadd and fmul (update if data_t changes)
    aggregate_t rowAggregate[t], columnAggregate[2 * t - 1];
    // =============== [Scatter] ===============
    // RowStreaming: Read Values for the current row
    // TODO: Reformat memory writes
    data_t Ti = 0, mui = 0;
    RowLaneScatterRow:
    for (index_t i = 0; i < t; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < m) {
            Ti = Ti_in.read();
            Ti_m[i] = Ti;
        }

        if (i == 0) {
            mui = mui_in.read();
            mui_m = mui;
        }

        // Assumption t >= m, therefore no condition here
        data_t dfi = dfi_in.read();
        dfi_m[i] = dfi;

        data_t dgi = dgi_in.read();
        dgi_m[i] = dgi;

        data_t invi = invi_in.read();
        invi_m[i] = invi;

        rowAggregate[i] = aggregate_t_init;

        // check that we are not the last element
        // if not we can forward elements to the remaining rowElements
        if (xStage < nTiles - 1) {
            if (i < m)
                Ti_out.write(Ti);
            if (i == 0)
                mui_out.write(mui);
            dfi_out.write(dfi);
            dgi_out.write(dgi);
            invi_out.write(invi);
        }
    }

    // Column Streaming: Read Values for the current column
    data_t muj = 0, dfj = 0, dgj = 0, invj = 0;
    RowLaneScatterColumn:
    for (index_t i = 0; i < (n - t * xStage); ++i) {
        #pragma HLS PIPELINE II=1
        data_t Tj = Tj_in.read();

        if (i < (n - m + 1 - t * xStage)) {
            muj = muj_in.read();
            dfj = dfj_in.read();
            dgj = dgj_in.read();
            invj = invj_in.read();
        }

        if (i < t + m - 1)
            Tj_m[i] = Tj;

        if (i < t)
            muj_m[i] = muj;

        if (i < 2 * t - 1) {
            dfj_m[i] = dfj;
            dgj_m[i] = dgj;
            invj_m[i] = invj;
            columnAggregate[i] = aggregate_t_init;
        }

        // do not forward first t elements (only concern current element)
        if (i >= t) {
            Tj_out.write(Tj);
            if (i < (n - m + 1 - t * xStage)) {
                muj_out.write(muj);
                dfj_out.write(dfj);
                dgj_out.write(dgj);
                invj_out.write(invj);
            }
        }

    }
    // =============== [/Scatter] ===============

    // =============== [Compute] ===============

    data_t QT[t], P[t];
    aggregate_t rowAggregate_m = aggregate_t_init;
    
    // Compute the Matrix Profile (here)
    const index_t yOffset = yStage * t;
    const index_t xOffset = xStage * t;

    // Exclusion Zone <==> realX - m/4 <= realY <= realX + m/4
    // 				  <==> (xStage * t + r + i) - m/4 <= yStage * t + r <= (xStage * t + r + i) + m/4
    // 				  <==> xStage * t + i - m/4 <= yStage * t <= xStage * t + i + m/4
    // 				  <==> xStage * t + i - m/4 <= yStage * t [t > 0, xStage >= yStage, i > 0, m/4 > 0]
    //    			  <==> (xStage - yStage) * t + i - m/4 <= 0
    // 				  <==> (xStage - yStage) * t - m/4 <= -i
    //   			  <==> i <= (yStage - xStage) * t + m/4
    const index_t exclusionZone = max((yStage - xStage) * t + m / 4, 0);

    // compute the first row of the matrix
    MatrixProfileComputeInitQRow:
    for (index_t i = exclusionZone; i < min(t, n - m + 1 - t * xStage); ++i) {
        // compute convolution explicitly
        data_t sum = 0;
        MatrixProfileComputeInitQColumn:
        for (index_t j = 0; j < m; j++) {
            sum += (Ti_m[j] - mui_m) * (Tj_m[i + j] - muj_m[i]);
        }
        QT[i] = sum;
        data_t PearsonCorrelation = QT[i] * invi_m[0] * invj_m[i];

        // update aggregates in case of improvement
        if (PearsonCorrelation > rowAggregate_m.value)
            rowAggregate_m = {PearsonCorrelation, static_cast<index_t>(i + xOffset)}; // remember "best" column for rows
        if (PearsonCorrelation != aggregate_init)
            columnAggregate[i] = {PearsonCorrelation, static_cast<index_t>(0 + yOffset)}; // remember "best" row for columns
    }
    rowAggregate[0] = rowAggregate_m;    

    // Compute (t-1) rows using the simplification
    MatrixProfileComputeRow:
    for (index_t r = 1; r < t; ++r) {
        rowAggregate_m = aggregate_t_init;
        MatrixProfileComputeColumn:
        for (index_t i = exclusionZone; i < min(t, n - m - t * xStage - r + 1); ++i) {
            #pragma HLS PIPELINE II=1
            
            // QT_{i, j} = QT_{i-1, j-1} + df_i * dg_j + df_j * dg_i
            // QT[i] was the previous value (i.e. value diagonally above the current QT[i])
            QT[i] = QT[i] + dfi_m[r] * dgj_m[i + r] + dfj_m[i + r] * dgi_m[r];

            // calculate Pearson Correlation
            P[i] = QT[i] * invi_m[r] * invj_m[i + r];
            
            if (P[i] > rowAggregate_m.value)
                rowAggregate_m = {P[i], static_cast<index_t>(r + i + xOffset)}; // remember "best" column for rows
            
            const index_t column = r + i;
            if (P[i] > columnAggregate[column].value)
                columnAggregate[column] = {P[i], static_cast<index_t>(r + yOffset)}; // remember "best" row for columns
        }
        rowAggregate[r] = rowAggregate_m;
    }

    // =============== [/Compute] ===============

    // =============== [Reduce] ===============
    // Just rows
    RowLaneReduceRow:
    for (index_t i = 0; i < t; ++i) {
        #pragma HLS PIPELINE II=1
        aggregate_t prevRowAggregate = rowAggregate_in.read();
        aggregate_t curRowAggregate  = rowAggregate[i];
        rowAggregate_out.write((curRowAggregate.value > prevRowAggregate.value) ? curRowAggregate : prevRowAggregate);
    }

    // Just columns
    data_t columnAggregateLength = min((xStage - yStage) * t + t - 1 + t, n - m + 1 - t * yStage);
    RowLaneReduceColumn:
    for (index_t i = 0; i < columnAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        aggregate_t prevColAggregate = (i < (xStage - yStage) * t + t - 1) ? columnAggregate_in.read() : aggregate_t_init;
        aggregate_t curColAggregate  = (i >= (xStage - yStage) * t) ? columnAggregate[i - (xStage - yStage) * t] : aggregate_t_init;
        columnAggregate_out.write((curColAggregate.value > prevColAggregate.value) ? curColAggregate : prevColAggregate);
    }
    // =============== [/Reduce] ===============
}

void ReductionElement(index_t yStage, stream<aggregate_t, stream_d> &rRow_in, stream<aggregate_t, stream_d> &rCol_in,
                      stream<aggregate_t, stream_d> &rowAggregate_in, stream<aggregate_t, stream_d> &columnAggregate_in,
                      stream<aggregate_t, stream_d> &rRow_out, stream<aggregate_t, stream_d> &rCol_out) {
    // forward row aggregates
    index_t rowAggregateLength = min(t * (yStage + 1), n - m + 1);
    ReductionReduceRow:
    for (index_t i = 0; i < rowAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        rRow_out.write((i < t * yStage) ? rRow_in.read() 
                                        : rowAggregate_in.read());
    }

    // forward columns & reduce with previous element in case of contention
    index_t columnAggregateLength = n - m + 1;
    ReductionReduceColumn:
    for (index_t i = 0; i < columnAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        // Previous Aggregates
        aggregate_t prevColAggregate = yStage > 0 ? rCol_in.read() 
                                                  : aggregate_t_init;
        // Aggregates of the current row
        aggregate_t colAggregate = (i >= t * yStage) ? columnAggregate_in.read() 
                                                     : aggregate_t_init;
        // Reduce Aggregates using max
        rCol_out.write((colAggregate.value > prevColAggregate.value) ? colAggregate : prevColAggregate);
    }
}

data_t PearsonCorrelationToEuclideanDistance(data_t PearsonCorrelation) {
    #pragma HLS INLINE
    return sqrt(static_cast<data_t>(2 * m * (1 - PearsonCorrelation)));
}

void StreamToMemoryElement(stream<aggregate_t, stream_d> &rRow_in, stream<aggregate_t, stream_d> &rCol_in, data_t *MP, index_t *MPI) {
    // local "cache" storing the row aggregates (to merge them later)
    aggregate_t aggregates_m[n - m + 1];

    // read the row-wise aggregates from reduction-lane
    StreamToMemoryReduceRows:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPLEINE II=1
        aggregates_m[i] = rRow_in.read();
    }

    // read the column-wise aggregates from reduction-lane, reduce them with
    // row-wise aggregates and calculate PearsonCorrelation
    StreamToMemoryReduceColumns:
    for (index_t i = 0; i < n - m + 1; ++i) {
        aggregate_t rowAggregate = aggregates_m[i];
        aggregate_t columnAggregate = rCol_in.read();
        MP[i] = PearsonCorrelationToEuclideanDistance(rowAggregate.value > columnAggregate.value ? rowAggregate.value : columnAggregate.value);
        MPI[i] = rowAggregate.value > columnAggregate.value ? rowAggregate.index : columnAggregate.index;
    }
}

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=T   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem2
    
    #pragma HLS DATAFLOW

    constexpr index_t nStreams = ((nTiles + 1) * nTiles) / 2;

    // Streams for the Scatter Lane
    stream<data_t, stream_d> sT[nTiles + 1], sMu[nTiles + 1], sDf[nTiles + 1], sDg[nTiles + 1], sInv[nTiles + 1];

    // Streams for rows (i.e. not scatter lane)
    stream<data_t, stream_d> Ti[nStreams], Tj[nStreams], mui[nStreams], muj[nStreams];
    stream<data_t, stream_d> dfi[nStreams], dfj[nStreams], dgi[nStreams], dgj[nStreams], invi[nStreams], invj[nStreams];

    // Streams for rows (aggregates)
    stream<aggregate_t, stream_d> rowAggregate[nStreams], columnAggregate[nStreams];

    // Streams for the Reduction Lane
    stream<aggregate_t, stream_d> rRow[nTiles + 1], rCol[nTiles + 1];

    MemoryToStreamElement(T, sT[0], sMu[0], sDf[0], sDg[0], sInv[0]);

    #ifdef TEST_MOCK_SW
    for (index_t y = 0; y < nTiles; ++y) {
        #pragma HLS UNROLL
        const index_t beginIndex = index2D(y, y);
        PrimaryDiagonalComputeElement(y, y, sT[y], sMu[y], sDf[y], sDg[y], sInv[y], Ti[beginIndex], Tj[beginIndex],
                mui[beginIndex], muj[beginIndex], dfi[beginIndex], dfj[beginIndex], dgi[beginIndex],
                dgj[beginIndex], invi[beginIndex], invj[beginIndex], rowAggregate[beginIndex], columnAggregate[beginIndex],
                sT[y + 1], sMu[y + 1], sDf[y + 1], sDg[y + 1], sInv[y + 1]);

        for (index_t x = y + 1; x < nTiles; ++x) {
            #pragma HLS UNROLL
            index_t index = index2D(x, y);
            DiagonalComputeElement(y, x, Ti[index - 1], Tj[index - 1], mui[index - 1], muj[index - 1],
                    dfi[index - 1], dfj[index - 1], dgi[index - 1], dgj[index - 1], invi[index - 1],
                    invj[index - 1], rowAggregate[index - 1], columnAggregate[index - 1], Ti[index],
                    Tj[index], mui[index], muj[index], dfi[index], dfj[index], dgi[index], dgj[index], 
                    invi[index], invj[index], rowAggregate[index], columnAggregate[index]);
        }
        const index_t endIndex = index2D(nTiles - 1, y);
        ReductionElement(y, rRow[y], rCol[y], rowAggregate[endIndex], columnAggregate[endIndex], rRow[y + 1], rCol[y + 1]);
    }
    #else
        #include "kernel/Stream2DInit.hpp"
    #endif

    StreamToMemoryElement(rRow[nTiles], rCol[nTiles], MP, MPI);
}
