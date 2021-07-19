#include "libspir_types.h"
#include "hls_stream.h"
#include "xcl_top_defines.h"
#include "ap_axi_sdata.h"
#define EXPORT_PIPE_SYMBOLS 1
#include "cpu_pipes.h"
#undef EXPORT_PIPE_SYMBOLS
#include "xcl_half.h"
#include <cstddef>
#include <vector>
#include <complex>
#include <pthread.h>
using namespace std;

extern "C" {

void MatrixProfileKernelTLF(size_t QTInit, size_t data, size_t MP, size_t MPI);

static pthread_mutex_t __xlnx_cl_MatrixProfileKernelTLF_mutex = PTHREAD_MUTEX_INITIALIZER;
void __stub____xlnx_cl_MatrixProfileKernelTLF(char **argv) {
  void **args = (void **)argv;
  size_t QTInit = *((size_t*)args[0+1]);
  size_t data = *((size_t*)args[1+1]);
  size_t MP = *((size_t*)args[2+1]);
  size_t MPI = *((size_t*)args[3+1]);
 pthread_mutex_lock(&__xlnx_cl_MatrixProfileKernelTLF_mutex);
  MatrixProfileKernelTLF(QTInit, data, MP, MPI);
  pthread_mutex_unlock(&__xlnx_cl_MatrixProfileKernelTLF_mutex);
}
}
