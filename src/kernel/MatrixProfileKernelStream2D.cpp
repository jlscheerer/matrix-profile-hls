/**
 * @file    MatrixProfileKernelStream2D.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Stream-2D]
 */
// TODO: Maybe move ProcessingElements into individual files
#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_math.h"
#include "hls_stream.h"

// "tile-size"
constexpr size_t t = 4;

using hls::stream;
constexpr size_t stream_d = 3;

// TODO: Just define constexpr constant for the ceilDiv result
constexpr size_t ceilDiv(const size_t x, const size_t y) { return (x + y - 1) / y; }
constexpr size_t min(const size_t a, const size_t b){ return (a < b) ? a : b; }

void MemoryToStream(const data_t *T, stream<data_t, stream_d> &sT, stream<data_t, stream_d> &sMu,
                    stream<data_t, stream_d> &sDf, stream<data_t, stream_d> &sDg, stream<data_t, stream_d> &sInv) {
    // store the previous (m-1) T-values in local "cache" (acts as shift-register)
    data_t T_m[m];
    #pragma HLS ARRAY_PARITION variable=T_m complete

    // initialize mean calculation and local "cache"
    // read T values cannot be directly passed on (this would cause a deadlock in the ProcessingElements)
    PrecomputationInitT:
    for (size_t i = 0; i < m; ++i) {
        #pragma HLS PIPELINE II=1
        T_m[i] = T[i];
    }

    data_t mean = 0;
    PrecomputationInitMu:
    for (size_t i = 0; i < m; ++i){
        #pragma HLS UNROLL
        mean += T_m[i];
    }
    mean /= m;

    data_t inv_sum = 0;
    PrecomputationInitInv:
    for (size_t k = 0; k < m; ++k){
        #pragma HLS UNROLL
        inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
    }

    // do the first iteration manually (mean, inv, df, dg)
    sMu.write(mean); sDf.write(0); sDg.write(0);
    sInv.write(1 / sqrt(inv_sum));

    PrecomputationCompute:
    for (size_t i = m; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        data_t Ti = T[i];
        data_t Tm = T_m[0];

        // recompute mean to achieve II=1
        mean = 0;
        PrecomputationComputeUpdateMean:
        for(size_t k = 1; k < m; ++k) {
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
        for (size_t k = 1; k < m; ++k){
            #pragma HLS UNROLL
            inv_sum += (T_m[k] - mean) * (T_m[k] - mean);
        }
        // perform last element of the loop separately (this requires the new val
        inv_sum += (Ti - mean) * (Ti - mean);
        sInv.write(1 / sqrt(inv_sum));

        // shift all values in T_m back
        PrecomputationComputeShift:
        for (size_t k = 0; k < m - 1; ++k){
            #pragma HLS UNROLL
            T_m[k] = T_m[k + 1];
        }
        T_m[m - 1] = Ti;
    }

    // Push the remaining values
    PrecomputationScatterCacheT:
    for (size_t i = 0; i < m; ++i) {
        #pragma HLS PIPELINE II=1
        sT.write(T_m[i]);
    }
}

void MatrixProfileComputeUnit(size_t yStage, size_t xStage, data_t (&Ti_m)[m], data_t (&Tj_m)[t + m - 1], data_t mui_m, data_t (&muj_m)[t], 
                              data_t (&dfi_m)[t], data_t (&dfj_m)[2 * t - 1], data_t (&dgi_m)[t], data_t (&dgj_m)[2 * t - 1], data_t (&invi_m)[t], 
                              data_t (&invj_m)[2 * t - 1], data_t (&QT)[t], aggregate_t (&rowAggregate)[t], aggregate_t (&columnAggregate)[2 * t - 1]) {
    #pragma HLS INLINE
    // Compute the Matrix Profile (here)
    size_t yOffset = yStage * t;
    size_t xOffset = xStage * t;

    // Compute the first row of the matrix
    // TODO: Set row and column aggregate for this loop
    MatrixProfileComputeInitQRow:
    for (size_t i = 0; i < t; ++i) {
        data_t sum = 0;
        MatrixProfileComputeInitQColumn:
        for (size_t j = 0; j < m; j++) {
            sum += (Ti_m[j] - mui_m) * (Tj_m[i + j] - muj_m[i]);
        }
        QT[i] = sum;
        // See Proof Related to exclusion zone below
        // TODO: Move ExclusionZone and OutOfBounds into custom functions
        // r = 0
        bool exclusionZone = (i <= (yStage - xStage) * t + m / 4);
        bool outOfBounds = (i > n - m - t * xStage);
        data_t PearsonCorrelation = (!exclusionZone && !outOfBounds) ? QT[i] * invi_m[0] * invj_m[i]
                                                                     : aggregate_init;
        if (PearsonCorrelation > rowAggregate[0].value) {
            rowAggregate[0] = {PearsonCorrelation, static_cast<index_t>(i + xOffset)}; // remember "best" column for rows
        }
        if (PearsonCorrelation != aggregate_init) {
            columnAggregate[i] = {PearsonCorrelation, static_cast<index_t>(0 + yOffset)}; // remember "best" row for columns
        }
    }

    // TODO: Heavily optimize this Loop
    // Compute (t-1) rows using the simplification
    MatrixProfileComputeRow:
    for (size_t r = 1; r < t; ++r) {
        // TODO: Pull dfi/dgi out of the loop like in the naive example
        // TODO: Pull invi out of the loop like in the naive example
        MatrixProfileComputeColumn:
        for (size_t i = 0; i < t; ++i) {
            // by design it will always hold that realX >= realY
            // int realY = (yStage * t + r);
            // int realX = (xStage * t + r + i);

            // QT[i] = QT[i] + df[i] * dg[j] + df[j] * dg[i]
            QT[i] = QT[i] + dfi_m[r] * dgj_m[i + r] + dfj_m[i + r] * dgi_m[r];
            // Calculate Pearson Correlation and Include Exclusion zone
            // Exclusion Zone <==> realX - m/4 <= realY <= realX + m/4
            // 				  <==> (xStage * t + r + i) - m/4 <= yStage * t + r <= (xStage * t + r + i) + m/4
            // 				  <==> xStage * t + i - m/4 <= yStage * t <= xStage * t + i + m/4
            // 				  <==> xStage * t + i - m/4 <= yStage * t [t > 0, xStage >= yStage, i > 0, m/4 > 0]
            //    			  <==> (xStage - yStage) * t + i - m/4 <= 0
            // 				  <==> (xStage - yStage) * t - m/4 <= -i
            //   			  <==> i <= (yStage - xStage) * t + m/4
            // TODO: Move ExclusionZone and OutOfBounds into custom functions
            bool exclusionZone = (i <= (yStage - xStage) * t + m / 4);
            bool outOfBounds = (i > n - m - t * xStage - r);
            data_t PearsonCorrelation = (!exclusionZone && !outOfBounds) ? QT[i] * invi_m[r] * invj_m[i + r]
                                                                         : aggregate_init;
            if (PearsonCorrelation > rowAggregate[r].value) {
                rowAggregate[r] = {PearsonCorrelation, static_cast<index_t>(r + i + xOffset)}; // remember "best" column for rows
            }
            if (PearsonCorrelation > columnAggregate[r + i].value) {
                columnAggregate[r + i] = {PearsonCorrelation, static_cast<index_t>(r + yOffset)}; // remember "best" row for columns
            }
        }
    }

}

void ScatterLaneStreamingUnit(size_t yStage, size_t xStage, stream<data_t, stream_d> &sT_in,
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

    // =============== [Scatter] ===============
    data_t mu = 0, df = 0, dg = 0, inv = 0;
    ScatterDiagonalLane:
    for (size_t i = 0; i < n - t * xStage; ++i) {
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
        }

        if (i < 2 * t - 1) {
            dfj_m[i] = df;
            dgj_m[i] = dg;
            invj_m[i] = inv;
        }

        if (xStage < ceilDiv(n - m + 1, t) - 1) {
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
    // TODO: (Move up +) Initialize during scatter
    data_t QT[t];
    
    aggregate_t rowAggregate[t];
    ScatterAggregateInitRow:
    for (size_t i = 0; i < t; ++i){
        #pragma HLS PIPELINE II=1
        rowAggregate[i] = aggregate_t_init;
    }

    aggregate_t columnAggregate[2 * t - 1];
    ScatterAggregateInitColumn:
    for (size_t i = 0; i < 2 * t - 1; ++i){
        #pragma HLS PIPELINE II=1
        columnAggregate[i] = aggregate_t_init;
    }

    // After getting the required data can now compute portion of the matrix profile
    MatrixProfileComputeUnit(yStage, xStage, Ti_m, Tj_m, mui_m, muj_m, dfi_m, dfj_m, dgi_m, dgj_m, invi_m, invj_m, QT, rowAggregate, columnAggregate);
    // =============== [/Compute] ===============

    // =============== [Reduce] ===============
    // Don't need to forward anything because we are the first element of the row
    // Just row
    size_t rowAggregateLength = min(t, n - m + 1 - t * yStage);
    ScatterReduceAggregateRow:
    for (size_t i = 0; i < rowAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        rowAggregate_out.write(rowAggregate[i]);
    }
    // Just column
    size_t columnAggregateLength = min(t + (t - 1), n - m + 1 - t * yStage);
    ScatterReduceAggregateColumn:
    for (size_t i = 0; i < columnAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        columnAggregate_out.write(columnAggregate[i]);
    }
    // =============== [/Reduce] ===============
}

void RowLaneStreamingUnit(size_t yStage, size_t xStage, stream<data_t, stream_d> &Ti_in,
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

    // =============== [Scatter] ===============
    // RowStreaming: Read Values for the current row
    // TODO: Reformat memory writes
    data_t Ti = 0, mui = 0;
    RowLaneScatterRow:
    for (size_t i = 0; i < t; ++i) {
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

        // check that we are not the last element
        // if not we can forward elements to the remaining rowElements
        if (xStage < ceilDiv(n - m + 1, t) - 1) {
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
    for (size_t i = 0; i < (n - t * xStage); ++i) {
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

    // TODO: (Move up +) Initialize during scatter
    data_t QT[t];
    
    aggregate_t rowAggregate[t];
    RowLaneAggregateInitRow:
    for (size_t i = 0; i < t; ++i){
        #pragma HLS PIPELINE II=1
        rowAggregate[i] = aggregate_t_init;
    }

    aggregate_t columnAggregate[2 * t - 1];
    RowLaneAggregateInitColumn:
    for (size_t i = 0; i < 2 * t - 1; ++i){
        #pragma HLS PIPELINE II=1
        columnAggregate[i] = aggregate_t_init;
    }

    // After getting the required data can now compute portion of the matrix profile
    MatrixProfileComputeUnit(yStage, xStage, Ti_m, Tj_m, mui_m, muj_m, dfi_m, dfj_m, dgi_m, dgj_m, invi_m, invj_m, QT, rowAggregate, columnAggregate);
    // =============== [/Compute] ===============

    // =============== [Reduce] ===============
    // Just rows
    RowLaneReduceRow:
    for (size_t i = 0; i < t; ++i) {
        #pragma HLS PIPELINE II=1
        aggregate_t rowAgr = rowAggregate_in.read();
        // Reduce with current element
        // TODO: Make this a single read from rowAggregate[i]
        if (rowAggregate[i].value > rowAgr.value) {
            // forward our value
            rowAggregate_out.write(rowAggregate[i]);
        } else {
            // take previous value
            rowAggregate_out.write(rowAgr);
        }
    }

    // Just columns
    data_t columnAggregateLength = min((xStage - yStage) * t + t - 1 + t, n - m + 1 - t * yStage);
    RowLaneReduceColumn:
    for (size_t i = 0; i < columnAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        aggregate_t prevColAggregate = (i < (xStage - yStage) * t + t - 1) ? columnAggregate_in.read()
                                                                           : aggregate_t_init;

        aggregate_t colAggregate     = (i >= (xStage - yStage) * t) ? columnAggregate[i - (xStage - yStage) * t]
                                                                    : aggregate_t_init;

        if (colAggregate.value > prevColAggregate.value) {
            // forward our value
            columnAggregate_out.write(colAggregate);
        } else {
            // take previous value
            columnAggregate_out.write(prevColAggregate);
        }
    }
    // =============== [/Reduce] ===============
}

void RowReductionUnit(size_t yStage, stream<aggregate_t, stream_d> &rRow_in, stream<aggregate_t, stream_d> &rCol_in,
                      stream<aggregate_t, stream_d> &rowAggregate_in, stream<aggregate_t, stream_d> &columnAggregate_in,
                      stream<aggregate_t, stream_d> &rRow_out, stream<aggregate_t, stream_d> &rCol_out) {
    // forward row aggregates
    size_t rowAggregateLength = min(t * (yStage + 1), n - m + 1);
    ReductionReduceRow:
    for (size_t i = 0; i < rowAggregateLength; ++i) {
        #pragma HLS PIPELINE II=1
        rRow_out.write((i < t * yStage) ? rRow_in.read() 
                                        : rowAggregate_in.read());
    }

    // forward columns & reduce with previous element in case of contention
    size_t columnAggregateLength = n - m + 1;
    ReductionReduceColumn:
    for (size_t i = 0; i < columnAggregateLength; ++i) {
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
    return sqrt(2 * m * (1 - PearsonCorrelation));
}

void StreamToMemory(stream<aggregate_t, stream_d> &rRow_in, stream<aggregate_t, stream_d> &rCol_in, data_t *MP, index_t *MPI) {
    // local "cache" storing the row aggregates (to merge them later)
    aggregate_t aggregates_m[n - m + 1];

    // read the row-wise aggregates from reduction-lane
    StreamToMemoryReduceRows:
    for (size_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPLEINE II=1
        aggregates_m[i] = rRow_in.read();
    }

    // read the column-wise aggregates from reduction-lane, reduce them with
    // row-wise aggregates and calculate PearsonCorrelation
    StreamToMemoryReduceColumns:
    for (size_t i = 0; i < n - m + 1; ++i) {
        aggregate_t rowAggregate = aggregates_m[i];
        aggregate_t columnAggregate = rCol_in.read();
        MP[i] = PearsonCorrelationToEuclideanDistance(rowAggregate.value > columnAggregate.value ? rowAggregate.value : columnAggregate.value);
        MPI[i] = rowAggregate.value > columnAggregate.value ? rowAggregate.index : columnAggregate.index;
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

    constexpr size_t nTiles = ceilDiv(n - m + 1, t);
    constexpr size_t nStreams = ((nTiles + 1) * nTiles) / 2;

    // Streams for the Scatter Lane
    stream<data_t, stream_d> sT[nTiles + 1], sMu[nTiles + 1], sDf[nTiles + 1], sDg[nTiles + 1], sInv[nTiles + 1];

    // Streams for rows (i.e. not scatter lane)
    stream<data_t, stream_d> Ti[nStreams], Tj[nStreams], mui[nStreams], muj[nStreams];
    stream<data_t, stream_d> dfi[nStreams], dfj[nStreams], dgi[nStreams], dgj[nStreams], invi[nStreams], invj[nStreams];

    // Streams for rows (aggregates)
    stream<aggregate_t, stream_d> rowAggregate[nStreams], columnAggregate[nStreams];

    // Streams for the Reduction Lane
    stream<aggregate_t, stream_d> rRow[nTiles + 1], rCol[nTiles + 1];

    MemoryToStream(T, sT[0], sMu[0], sDf[0], sDg[0], sInv[0]);

    // https://math.stackexchange.com/questions/2134011/conversion-of-upper-triangle-linear-index-from-index-on-symmetrical-array
    // TODO: Potentially Flatten this Loop (OR UNROLL twice?)
    for (size_t y = 0; y < nTiles; ++y) {
        #pragma HLS UNROLL
        // TODO: Move Index into constexpr method
        // x == y
        index_t beginIndex = (nTiles * (nTiles - 1)) / 2 - ((nTiles - y) * (nTiles - y - 1)) / 2 + y;
        ScatterLaneStreamingUnit(y, y, sT[y], sMu[y], sDf[y], sDg[y], sInv[y], Ti[beginIndex], Tj[beginIndex],
                mui[beginIndex], muj[beginIndex], dfi[beginIndex], dfj[beginIndex], dgi[beginIndex],
                dgj[beginIndex], invi[beginIndex], invj[beginIndex], rowAggregate[beginIndex], columnAggregate[beginIndex],
                sT[y + 1], sMu[y + 1], sDf[y + 1], sDg[y + 1], sInv[y + 1]);

        for (size_t x = y + 1; x < nTiles; ++x) {
            #pragma HLS UNROLL
            index_t index = (nTiles * (nTiles - 1)) / 2 - ((nTiles - y) * (nTiles - y - 1)) / 2 + x;
            RowLaneStreamingUnit(y, x, Ti[index - 1], Tj[index - 1], mui[index - 1], muj[index - 1],
                    dfi[index - 1], dfj[index - 1], dgi[index - 1], dgj[index - 1], invi[index - 1],
                    invj[index - 1], rowAggregate[index - 1], columnAggregate[index - 1], Ti[index],
                    Tj[index], mui[index], muj[index], dfi[index], dfj[index], dgi[index], dgj[index], 
                    invi[index], invj[index], rowAggregate[index], columnAggregate[index]);
        }
        // x == nTiles - 1
        index_t endIndex = (nTiles * (nTiles - 1)) / 2 - ((nTiles - y) * (nTiles - y - 1)) / 2 + (nTiles - 1);
        RowReductionUnit(y, rRow[y], rCol[y], rowAggregate[endIndex], columnAggregate[endIndex], rRow[y + 1], rCol[y + 1]);
    }

    StreamToMemory(rRow[nTiles], rCol[nTiles], MP, MPI);
}
