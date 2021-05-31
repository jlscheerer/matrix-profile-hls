#include "gtest/gtest.h"

#include <cstdlib>
#include <cmath>

// wrapper struct to have multiple StreamlessKernel instances
// with different confugrations (data_t, index_t, n, m)
// this allows for multiple test (without having to recompile)
template<typename data_t, typename index_t, size_t n, size_t m>
struct StreamlessKernel {
    #include "MatrixProfileKernelStreamless.cpp"
};

TEST(TestStreamlessKernel, TestSmall8Syn) {
    EXPECT_EQ(0, 0);
}

TEST(TestStreamlessKernel, TestSmall16Syn) {
    EXPECT_EQ(0, 0);
}