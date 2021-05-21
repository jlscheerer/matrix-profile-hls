/**
 * @file    MatrixProfileKernel.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Declaration of the MatrixProfileKernel Signature
 */

#pragma once
#include <cstdlib>

extern "C" {

    void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI);

}
