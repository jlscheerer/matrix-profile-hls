/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    
    #include "kernel/MatrixProfileKernel.hpp"
    #include "kernel/TreeReduce.hpp"
    
    #include "hls_math.h"
#endif

struct ComputePack { data_t df, dg, inv; };

void MatrixProfileKernelTLF(const InputDataPack *in, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MPI offset=slave bundle=gmem1

    data_t QT[n - m + 1];
    ComputePack columnData[n - m + 1]; aggregate_t columnAggregate[n - m + 1];
    ComputePack rowData[n - m + 1]; aggregate_t rowAggregate[n - m + 1];

    MatrixProfileInit:
    for (index_t i = 0; i < n - m + 1; ++i) {
       	#pragma HLS PIPELINE II=1
        const InputDataPack read = in[i];
        const ComputePack compute = ComputePack{read.df, read.dg, read.inv};
        QT[i] = read.QT;
        rowData[i] = compute;
        columnData[i] = compute;
    }

    aggregate_t rowReduce[16];
    #pragma HLS ARRAY_PARTITION variable=rowReduce complete

    // rowReduce needs to be explicitly initialized if n - m + 1 < 16
    // Because for every row we perform a reduction on all elements!
    // For every n - m + 1 >= 16 we handle this via implicit initializaiton
    for (index_t i = 0; i < 16; ++i) {
	#pragma HLS UNROLL
	rowReduce[i] = aggregate_t_init;
    }

    constexpr int T = 4;
    aggregate_t columnReduce[n - m + 1][T];
    #pragma HLS ARRAY_PARTITION variable=T dim=2 complete

    // =============== [/Compute] ===============
    MatrixProfileComputeRow:
    for (index_t k = 0; k < n - m + 1; ++k) {
        MatrixProfileComputeColumn:
        for (index_t i = 0; i < n - m + 1; ++i) {
            #pragma HLS PIPELINE II=1
            const index_t columnIndex = k + i;
            const bool computationInRange = k + i < n - m + 1;
            const bool exclusionZone = i < (m / 4);

            const ComputePack row = rowData[k];
            const ComputePack column = (!exclusionZone && computationInRange)
                                            ? columnData[columnIndex] : (ComputePack){0, 0, 0};

	    QT[i] += row.df * column.dg + column.df * row.dg;

            // calculate pearson correlation
            // P_{i, j} = QT_{i, j} * inv_i * inv_j
            const data_t P = QT[i] * row.inv * column.inv;

            // Row-Wise Partial Reduction
            aggregate_t prevRow = (i < 16) ? aggregate_t_init : rowReduce[i % 16];
            rowReduce[i % 16] = P > prevRow.value ? aggregate_t(P, columnIndex) : prevRow;

            // Column-Wise Partial Reduction
            // Wrap-Around works because if we are not outside the exlusionZone value will be 0
            aggregate_t prevColumn = (k < T/2) ? aggregate_t_init : columnReduce[columnIndex % (n - m + 1)][k % T];
	    columnReduce[columnIndex % (n - m + 1)][(k + T/2) % T] = prevColumn.value > P ? prevColumn : aggregate_t(P, k);
        }

        rowAggregate[k] = TreeReduce::Maximum<aggregate_t, 16>(rowReduce);
    }
    // =============== [/Compute] ===============
    
    // =============== [Reduce] ===============
#if 0
    ReduceColumns:
    for (int i = 0; i < n - m + 1; ++i) {
	// needs to read four elements per iteration to reduce
	#pragma HLS PIPELINE II=1
    	columnAggregate[i] = TreeReduce::Maximum<aggregate_t, T>(columnReduce[i]);
    }
#endif

    // compute maximum between row- and column-wise aggregates
    ReductionCompute:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
	const aggregate_t rowAggregateV = rowAggregate[i];
	const aggregate_t columnAggregateV = TreeReduce::Maximum<aggregate_t, T>(columnReduce[i]);
        const aggregate_t aggregate = rowAggregateV.value > columnAggregateV.value
					? rowAggregateV : columnAggregateV;
        MP[i] = aggregate.value;
        MPI[i] = aggregate.index;
    }
    // =============== [/Reduce] ===============
}
