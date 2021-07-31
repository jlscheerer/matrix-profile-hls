/**
 * @file    TestStream1DKernel.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Software Tests for Stream1D-Kernel
 */

#include "MatrixProfileTests.hpp"

namespace MatrixProfileTests {

    // wrapper struct to have multiple Stream1DKernel instances
    // with different configurations (data_t, index_t, w) this allows
    // for multiple test (without having to recompile)
    template<typename data_t, typename index_t, int w, int t>
    struct Stream1DKernel: public MatrixProfileKernel<data_t, index_t, w> {
        
        using aggregate_t = typename MatrixProfileKernel<data_t, index_t, w>::aggregate_t;
        using InputDataPack = typename MatrixProfileKernel<data_t, index_t, w>::InputDataPack;
        using OutputDataPack = typename MatrixProfileKernel<data_t, index_t, w>::OutputDataPack;

        static constexpr data_t aggregate_init = MatrixProfileKernel<data_t, index_t, w>::aggregate_init;
        static constexpr index_t index_init = MatrixProfileKernel<data_t, index_t, w>::index_init;

        static constexpr index_t nColumns = MatrixProfileKernel<data_t, index_t, w>::nColumns;

        #include "MatrixProfileKernelStream1D.cpp"
    };

    TEST(TestStream1DKernel, TestSmall128SynM16) {
        Stream1DKernel<double, int, 32, 16> kernel;
        TestMatrixProfileKernel<double, int, 128, 16, 32>(kernel, "synthetic/small128_syn.txt");
    }

    TEST(TestStream1DKernel, TestBenchmark1024SynM16W32T16) {
        Stream1DKernel<double, int, 32, 16> kernel;
        TestMatrixProfileKernel<double, int, 1024, 16, 32>(kernel, "benchmark/1024.txt");
    }

    TEST(TestStream1DKernel, TestBenchmark1024SynM16W256T128) {
        Stream1DKernel<double, int, 256, 128> kernel;
        TestMatrixProfileKernel<double, int, 1024, 16, 256>(kernel, "benchmark/1024.txt");
    }

    TEST(TestStream1DKernel, TestBenchmark1024SynM32W256T128) {
        Stream1DKernel<double, int, 512, 128> kernel;
        TestMatrixProfileKernel<double, int, 1024, 32, 512>(kernel, "benchmark/1024.txt");
    }

    TEST(TestStream1DKernel, TestBenchmark16384SynM128W1024T128) {
        Stream1DKernel<double, int, 1024, 128> kernel;
        TestMatrixProfileKernel<double, int, 16384, 128, 1024>(kernel, "benchmark/16384.txt");
    }

}