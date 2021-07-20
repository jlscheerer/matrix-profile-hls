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

void MatrixProfileKernelTLF(const data_t *QTInit, const ComputePack *data, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=QTInit offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=MP     offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MPI    offset=slave bundle=gmem2

    data_t QT[n - m + 1], P[n - m + 1];
    ComputePack columnData[n - m + 1]; aggregate_t columnAggregate[n - m + 1];
    ComputePack rowData[n - m + 1]; aggregate_t rowAggregate[n - m + 1];

    // TODO: Could move to implicit instantialization of row- and columnAggregate during the computation
    MatrixProfileInit:
    for (index_t i = 0; i < n - m + 1; ++i) {
       	#pragma HLS PIPELINE II=1
        QT[i] = QTInit[i];
        rowData[i] = data[i]; rowAggregate[i] = aggregate_t_init;
        columnData[i] = data[i]; columnAggregate[i] = aggregate_t_init;
    }

    data_t rowReduce[16];
    #pragma HLS ARRAY_PARTITION variable=rowReduce complete
    // =============== [/Compute] ===============
    // Do the actual calculations via updates
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
            P[i] = QT[i] * row.inv * column.inv;

            data_t prev = (i == 0) ? aggregate_init : rowReduce[i % 16];
            rowReduce[i % 16] = prev > P[i] ? prev : P[i];
            // if (computationInRange)
            // columnAggregate[columnIndex + (k % 2 ? 0 : n - m + 1)] = nextColumn > prevColumn 
            //							? nextColumn : prevColumn;
            //const aggregate_t cValue{P, k}, rValue{P, columnIndex};
            
            //columnAggregate[columnIndex] = columnAggregate[columnIndex] > cValue 
            //					? columnAggregate[columnIndex] : cValue;
                //rowAggregate[k] = rowAggregate[k] > rValue ? rowAggregate[k] :  rValue;
	    }

        MP[k] = TreeReduce::Maximum<data_t, 16>(rowReduce);
    }
    // =============== [/Compute] ===============
    
    // =============== [Reduce] ===============
    // Just always take the max
    ReductionCompute:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        const aggregate_t aggregate = rowAggregate[i] > columnAggregate[i] 
					? rowAggregate[i] : columnAggregate[i];
        MP[i]  = QT[i]; MPI[i] = aggregate.index;
    }
    // =============== [/Reduce] ===============
}
