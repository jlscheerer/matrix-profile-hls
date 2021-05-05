#pragma once

extern "C" {

    void MatrixProfileKernelTLF(const unsigned int *in1, const unsigned int *in2, unsigned int *out_r, int size);

}