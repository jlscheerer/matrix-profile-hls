/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"
    #include "kernel/TreeReduce.h"

    #include "hls_math.h"
#endif

template<typename T>
T max(const T &&a, const T &&b) {
    #pragma HLS INLINE
    return a > b ? a : b;
}

template <typename T>
struct Max {
  template <typename T0, typename T1>
  static T Apply(T0 &&a, T1 &&b) {
    #pragma HLS INLINE
    return (a > b) ? a : b;
  }
  static constexpr T identity() { return std::numeric_limits<T>::min(); }
private:
  Max() = delete;
  ~Max() = delete;
};


void MatrixProfileKernelTLF(const data_t *QTInit, const ComputePack *data, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=QTInit offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=MP     offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MPI    offset=slave bundle=gmem2

    data_t QT[n - m + 1], P[n - m + 1];

    ComputePack rowData[n - m + 1], columnData[n - m + 1];

    MatrixProfileInitQT:
    for (index_t i = 0; i < n - m + 1; ++i) {
       	#pragma HLS PIPELINE
	QT[i] = QTInit[i];
	const ComputePack read = data[i];
	rowData[i] = read;
	columnData[i] = read;
    }

    aggregate_t rowAggregate[n - m + 1];
    aggregate_t columnAggregate[n - m + 1];

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

            const ComputePack row = rowData[k];
            const ComputePack column = computationInRange ? columnData[columnIndex] : (ComputePack){0, 0, 0};
            
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
	
	data_t m01 = rowReduce[0] > rowReduce[1] ? rowReduce[0] : rowReduce[1];
	data_t m23 = rowReduce[2] > rowReduce[3] ? rowReduce[2] : rowReduce[3];
	data_t m45 = rowReduce[4] > rowReduce[5] ? rowReduce[4] : rowReduce[5];
	data_t m67 = rowReduce[6] > rowReduce[7] ? rowReduce[6] : rowReduce[7];

	data_t m03 = m01 > m23 ? m01 : m23;
	data_t m47 = m45 > m67 ? m45 : m67;
	MP[k] = m03 > m47 ? m03 : m47;
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
