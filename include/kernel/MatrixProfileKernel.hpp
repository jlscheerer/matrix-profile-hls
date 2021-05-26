/**
 * @file    MatrixProfileKernel.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Declaration of the MatrixProfileKernel Signature
 */

#pragma once
#include <cstdlib>

typedef struct {
    data_t value;
    index_t index;
} aggregate_t;

constexpr aggregate_t aggregate_t_init{aggregate_init, index_init};

extern "C" {

    void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI);

}
