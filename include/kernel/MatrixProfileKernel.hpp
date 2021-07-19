/**
 * @file    MatrixProfileKernel.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Declaration of the MatrixProfileKernel Signature
 */

#pragma once

#include "Config.hpp"

struct aggregate_t { 
    data_t value; index_t index; 
    aggregate_t() = default;
    aggregate_t(const data_t value, const index_t index)
        : value(value), index(index) {}
    
    bool operator<(const data_t other) const { 
        #pragma HLS INLINE
        return value > other;
    }
    bool operator>(const aggregate_t other) const {
        #pragma HLS INLINE
        return value > other.value;
    }
};

static const aggregate_t aggregate_t_init{aggregate_init, index_init};

extern "C" {

    void MatrixProfileKernelTLF(const data_t *QTInit, const ComputePack *data, data_t *MP, index_t *MPI);

}
