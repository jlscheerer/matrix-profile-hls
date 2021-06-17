/**
 * @file    MatrixProfileHost.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Definition of the KernelTLF Name and Version Information
 */

#pragma once

#include "Config.hpp"

static const std::string KernelTLF{"MatrixProfileKernelTLF"};
static const std::string versionName{
    "\n****** Matrix Profile HLS (target = " + std::string{TARGET_TYPE_NAME} + ")\n  **** Host-Application (C++/OpenCL) (build " + std::string{__DATE__} + " " 
    + std::string{__TIME__} + ")" + "\n    ** Kernel (" + KernelTLF + ") [" + KERNEL_IMPL_NAME + "]\n"
    "     * Configuration (n = " + std::to_string(n) + ", m = " + std::to_string(m)
    #if KERNEL_IMPL_INDEX == 2
      + ", t = " + std::to_string(t)
    #endif
    + ") [data_t = " + DATA_TYPE_NAME + ", index_t = " + INDEX_TYPE_NAME + "]\n"
};
