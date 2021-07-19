/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#if !defined(TEST_MOCK_SW)
    #include "Config.hpp"
    #include "kernel/MatrixProfileKernel.hpp"

    #include "hls_math.h"
#endif

void MatrixProfileKernelTLF(const data_t *QTInit, const ComputePack *data, data_t *MP, index_t *MPI) {
    #pragma HLS INTERFACE m_axi port=QTInit offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=MP     offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=MPI    offset=slave bundle=gmem2

    data_t QT[n - m + 1], P[n - m + 1];
    aggregate_t rowAggregate[n - m + 1], columnAggregate[n - m + 1];

    for (index_t i = 0; i < n - m + 1; ++i) {
        QT[i] = QTInit[i];
        rowAggregate[i] = aggregate_t_init;
        columnAggregate[i] = aggregate_t_init;
    }

    for (int i = 0; i < n - m + 1; ++i)
        std::cout << "inv: "  << data[i].inv << " ";
    std::cout << std::endl;

    // =============== [/Compute] ===============
    // Do the actual calculations via updates
    MatrixProfileComputeRow:
    for (index_t k = 0; k < n - m + 1; ++k) {
        // exclusionZone integrated into loop bounds
        // exclusionZone <==> row - m/4 <= column <= row + m/4
        //               <==> column <= row + m/4 [(row <= column, m > 0) ==> row - m/4 <= column]
        //               <==> row + k <= row + m/4
        //               <==> k <= m/4
        MatrixProfileComputeColumn:
        for (index_t i = (m / 4); i < n - m + 1; ++i) {
            const index_t column = k + i;
            const bool computationInRange = k + i < n - m + 1;
            const ComputePack rowData = data[k];
            const ComputePack columnData = computationInRange ? data[column] : (ComputePack){0, 0, 0};

            // QT_{i, j} = QT_{i-1, j-1} + df_i * dg_j + df_j * dg_i
            // QT[k] was the previous value (i.e. value diagonally above the current QT[k])
            QT[i] += rowData.df * columnData.dg + columnData.df * rowData.dg;
            
            // calculate pearson correlation
            // P_{i, j} = QT_{i, j} * inv_i * inv_j
            P[i] = QT[i] * rowData.inv * columnData.inv;

            // Update Aggregates
            if(computationInRange && P[i] > columnAggregate[column].value)
                columnAggregate[column] = {P[i], static_cast<index_t>(k)};
            if(computationInRange && P[i] > rowAggregate[k].value)
                rowAggregate[k] = {P[i], static_cast<index_t>(column)};
        }
    }
    // =============== [/Compute] ===============
    
    // =============== [Reduce] ===============
    // Just always take the max
    ReductionCompute:
    for (index_t i = 0; i < n - m + 1; ++i) {
        #pragma HLS PIPELINE II=1
        aggregate_t rowAggregate_m = rowAggregate[i], columnAggregate_m = columnAggregate[i];
        MP[i]  = rowAggregate_m.value > columnAggregate_m.value ? rowAggregate_m.value : columnAggregate_m.value;
        MPI[i] = rowAggregate_m.value > columnAggregate_m.value ? rowAggregate_m.index : columnAggregate_m.index;
    }
    // =============== [/Reduce] ===============
}