#include "MatrixProfileTests.hpp"

#include <array>

// wrapper struct to have multiple Stream1DKernel instances
// with different confugrations (data_t, index_t, n, m)
// this allows for multiple test (without having to recompile)
template<typename data_t, typename index_t, size_t n, size_t m>
struct Stream1DKernel {
    #include "MockConfig.hpp"
    #include "MatrixProfileKernelStream1D.cpp"
};

TEST(TestStream1DKernel, TestSmall8Syn) {
    using data_t = double; using index_t = int;
    constexpr size_t n = 8;
    constexpr size_t m = 5;
    std::array<data_t, n> T{1,4,9,16,25,36,49,64};
    std::array<data_t, n - m + 1> MP;
    std::array<index_t, n - m + 1> MPI; 
    Stream1DKernel<data_t, int, n, m>().MatrixProfileKernelTLF(T.data(), MP.data(), MPI.data());
}

TEST(TestStream1DKernel, TestSmall16Syn) {
    EXPECT_EQ(0, 0);
}