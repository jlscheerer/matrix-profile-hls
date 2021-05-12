#pragma once
#include <cstdlib>

extern "C" {

    void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI);

}
