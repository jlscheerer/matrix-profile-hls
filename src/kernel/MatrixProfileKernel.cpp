/**
 * @file    MatrixProfileKernel.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Includes the selected Implementation of the Kernel (C++/Vitis HLS)
 */

// Select the Kernel Implementation based on the KERNEL_IMPLEMENTATION 
// macro. Set this macro in "include/MatrixProfile.hpp"
#if KERNEL_IMPLENTATION == KERNEL_STREAMLESS
    #include "MatrixProfileKernelStreamless.cpp"
#elif KERNEL_IMPLEMENTATION == KERNEL_STREAM1D
    #include "MatrixProfileKernelStream1D.cpp"
#elif KERNEL_IMPLEMENTATION == KERNEL_STREAM2D
    #include "MatrixProfileKernelStream2D.cpp"
#else
    #error Invalid Kernel Implementation!
#endif