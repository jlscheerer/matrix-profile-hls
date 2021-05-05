#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_stream.h"

using hls::stream;

void MatrixProfileKernelTLF(const size_t n, const size_t m, const data_t *T, data_t *MP, index_t *MPI) {
    for(int i = 0; i < rs_len; ++i){
        MP[i] = T[i];
        MPI[i] = -1;
    }
}
