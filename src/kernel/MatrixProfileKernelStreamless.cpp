/**
 * @file    MatrixProfileKernelStreamless.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Implementation of the Kernel (C++/Vitis HLS) [Streamless]
 */

#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_math.h"

void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) {
    // TODO: Actually Implement Streamless Kernel
    for(int i = 0; i < rs_len; ++i){
        MP[i] = aggregate_init;
        MPI[i] = index_init;
    }
}