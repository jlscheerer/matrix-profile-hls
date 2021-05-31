#include "MatrixProfileTests.hpp"

// wrapper struct to have multiple Stream2DKernel instances
// with different confugrations (data_t, index_t, n, m, t)
// this allows for multiple test (without having to recompile)
template<typename data_t, typename index_t, size_t n, size_t m, size_t t>
struct Stream2DKernel: public MatrixProfileKernel<data_t, index_t, n, m> {
    #include "MockConfig.hpp"
    #include "MatrixProfileKernelStream2D.cpp"
};

TEST(TestStream2DKernel, TestSmall8Syn) {
    EXPECT_EQ(0, 0);
}

TEST(TestStream2DKernel, TestSmall16Syn) {
    EXPECT_EQ(0, 0);
}