/**
 * @file    TestVanillaKernel.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Software Tests for Vanilla-Kernel
 */

#include "MatrixProfileTests.hpp"

namespace MatrixProfileTests {

    // wrapper struct to have multiple VanillaKernel instances
    // with different configurations (data_t, index_t, w) this allows
    // for multiple test (without having to recompile)
    template<typename data_t, typename index_t, int w>
    struct VanillaKernel: public MatrixProfileKernel<data_t, index_t, w> {

        using aggregate_t = typename MatrixProfileKernel<data_t, index_t, w>::aggregate_t;
        using InputDataPack = typename MatrixProfileKernel<data_t, index_t, w>::InputDataPack;
        using OutputDataPack = typename MatrixProfileKernel<data_t, index_t, w>::OutputDataPack;

        static constexpr data_t aggregate_init = MatrixProfileKernel<data_t, index_t, w>::aggregate_init;
        static constexpr index_t index_init = MatrixProfileKernel<data_t, index_t, w>::index_init;

        static constexpr index_t nColumns = MatrixProfileKernel<data_t, index_t, w>::nColumns;

        #include "MatrixProfileKernelVanilla.cpp"
    };
    
    TEST(TestVanillaKernel, TestSmall128SynM16W32) {
        VanillaKernel<double, int, 32> kernel;
        TestMatrixProfileKernel<double, int, 128, 16, 32>(kernel, "test/small128_syn.txt");
    }

    TEST(TestVanillaKernel, TestBenchmark1024SynM16W32) {
        VanillaKernel<double, int, 32> kernel;
        TestMatrixProfileKernel<double, int, 1024, 16, 32>(kernel, "test/1024.txt");
    }
    
    TEST(TestVanillaKernel, TestBenchmark16384SynM128W1024) {
        VanillaKernel<double, int, 1024> kernel;
        TestMatrixProfileKernel<double, int, 16384, 128, 1024>(kernel, "test/16384.txt");
    }

}