/**
 * @file    MatrixProfileHost.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Definition of the KernelTLF Name and Version Information
 */

#pragma once

static const std::string KernelTLF{"MatrixProfileKernelTLF"};
static const std::string versionName{
    "\n****** Matrix Profile HLS\n  **** Host-Application (C++/OpenCL) (build " + std::string{__DATE__} + " " + std::string{__TIME__} + ")" + "\n    ** Kernel (" + KernelTLF + ") [v1D-Stream]"
};