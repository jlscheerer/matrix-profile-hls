/**
 * @file    MatrixProfileKernel.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Declaration of the MatrixProfileKernel Signature
 */

#pragma once

#include "Config.hpp"

extern "C" {

    void MatrixProfileKernelTLF(const index_t n, const index_t m, const index_t iteration, const InputDataPack *columns, const InputDataPack *rows, OutputDataPack *out);

}
