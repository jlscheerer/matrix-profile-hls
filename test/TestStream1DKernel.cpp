#include "MatrixProfileTests.hpp"

// wrapper struct to have multiple Stream1DKernel instances
// with different confugrations (data_t, index_t, n, m)
// this allows for multiple test (without having to recompile)
template<typename data_t, typename index_t, size_t n, size_t m>
struct Stream1DKernel: public MatrixProfileKernel<data_t, index_t, n, m> {
    #include "MockConfig.hpp"
    #include "MatrixProfileKernelStream1D.cpp"
};

TEST(TestStream1DKernel, TestSmall8SynM4) {
    Stream1DKernel<double, int, 8, 4> kernel;
    TestMatrixProfileKernel(kernel, "synthetic/small8_syn.txt");
}

TEST(TestStream1DKernel, TestSmall16SynM4) {
    Stream1DKernel<double, int, 16, 4> kernel;
    TestMatrixProfileKernel(kernel, "synthetic/small16_syn.txt");
}

TEST(TestStream1DKernel, TestSmall128SynM4) {
    Stream1DKernel<double, int, 128, 4> kernel;
    TestMatrixProfileKernel(kernel, "synthetic/small128_syn.txt");
}

TEST(TestStream1DKernel, TestBenchmark1024SynM4) {
    Stream1DKernel<double, int, 1024, 4> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}

TEST(TestStream1DKernel, TestBenchmark1024SynM20) {
    Stream1DKernel<double, int, 1024, 20> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}

TEST(TestStream1DKernel, TestBenchmark1024SynM4F) {
    Stream1DKernel<float, int, 1024, 4> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}

TEST(TestStream1DKernel, TestBenchmark1024SynM20F) {
    Stream1DKernel<float, int, 1024, 20> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}