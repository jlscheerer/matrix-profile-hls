#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

void MatrixProfileKernelTLF(const unsigned int *in1, const unsigned int *in2, unsigned int *out_r, int size) {
    // Local memory to store vector1
    unsigned int v1_buffer[BufferSize];

    // per iteration of this loop perform BUFFER_SIZE vector addition
    for (int i = 0; i < size; i += BufferSize) {
       #pragma HLS LOOP_TRIPCOUNT min=c_len max=c_len
        int chunk_size = BufferSize;

        // boundary checks
        if ((i + BufferSize) > size)
            chunk_size = size - i;

        read1: for (int j = 0; j < chunk_size; j++) {
           #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
            v1_buffer[j] = in1[i + j];
        }

        // burst reading B and calculating C and Burst writing to  Global memory
        vadd_writeC: for (int j = 0; j < chunk_size; j++) {
           #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
            // perform vector addition
            out_r[i+j] = v1_buffer[j] + in2[i+j];
        }
    }
}