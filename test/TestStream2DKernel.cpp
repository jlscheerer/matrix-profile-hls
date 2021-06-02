#include "MatrixProfileTests.hpp"

// wrapper struct to have multiple Stream2DKernel instances
// with different confugrations (data_t, index_t, n, m, t)
// this allows for multiple test (without having to recompile)
template<typename data_t, typename index_t, size_t n, size_t m, size_t t>
struct Stream2DKernel: public MatrixProfileKernel<data_t, index_t, n, m> {
    #include "MockConfig.hpp"
    #include "MatrixProfileKernelStream2D.cpp"
};
TEST(TestStream2DKernel, TestSmall8SynM4) {
    Stream2DKernel<double, int, 8, 4, 4> kernel;
    TestMatrixProfileKernel(kernel, "synthetic/small8_syn.txt");
}

TEST(TestStream2DKernel, TestSmall16SynM4) {
    Stream2DKernel<double, int, 16, 4, 8> kernel;
    TestMatrixProfileKernel(kernel, "synthetic/small16_syn.txt");
}

TEST(TestStream2DKernel, TestSmall128SynM4) {
    Stream2DKernel<double, int, 128, 4, 12> kernel;
    TestMatrixProfileKernel(kernel, "synthetic/small128_syn.txt");
}

TEST(TestStream2DKernel, TestBenchmark1024SynM4T20) {
    Stream2DKernel<double, int, 1024, 4, 20> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}

TEST(TestStream2DKernel, TestBenchmark1024SynM4T100) {
    Stream2DKernel<double, int, 1024, 4, 100> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}

TEST(TestStream2DKernel, TestBenchmark1024SynM20T500) {
    Stream2DKernel<double, int, 1024, 20, 500> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}

TEST(TestStream2DKernel, TestBenchmark1024SynM20T1000) {
    Stream2DKernel<double, int, 1024, 20, 1000> kernel;
    TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
}