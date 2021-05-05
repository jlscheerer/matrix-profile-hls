#pragma once

extern "C" {

    void MatrixProfileKernelTLF(const size_t n, const size_t m, const data_t *T, data_t *MP, index_t *MPI);

}
