#include "MatrixProfileTests.hpp"

// wrapper struct to have multiple Stream1DKernel instances
// with different confugrations (data_t, index_t, n, m)
// this allows for multiple test (without having to recompile)
template<typename data_t, typename index_t, size_t n, size_t m>
struct Stream1DKernel {
    #include "MockConfig.hpp"
    #include "MatrixProfileKernelStream1D.cpp"
};

TEST(TestStream1DKernel, TestSmall8Syn) {
    EXPECT_EQ(0, 0);
}

TEST(TestStream1DKernel, TestSmall16Syn) {
    EXPECT_EQ(0, 0);
}