#include "MatrixProfileTests.hpp"

// wrapper struct to have multiple Stream1DKernel instances
// with different confugrations (data_t, index_t, n, m)
// this allows for multiple test (without having to recompile)
template<typename data_t, typename index_t, size_t n, size_t m>
struct Stream1DKernel: public MatrixProfileKernel<data_t, index_t, n, m> {
    #include "MockConfig.hpp"
    #include "MatrixProfileKernelStream1D.cpp"
};

TEST(TestStream1DKernel, TestSmall8Syn) {
    Stream1DKernel<double, int, 8, 4> kernel;
    std::array<double, 8> T{1,4,9,16,25,36,49,64};
    TestMatrixProfileKernel(kernel, T);
}

TEST(TestStream1DKernel, TestSmall16Syn) {
    EXPECT_EQ(0, 0);
}