/**
 * @file    MatrixProfileKernel.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Declaration of the MatrixProfileKernel Signature
 */

#pragma once

#include "Config.hpp"

struct aggregate_t { data_t value; index_t index; };

static const aggregate_t aggregate_t_init{aggregate_init, index_init};

extern "C" {

    void MatrixProfileKernelTLF(const data_t *QTInit, const ComputePack *data, data_t *MP, index_t *MPI);

}
