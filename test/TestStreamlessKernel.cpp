/**
 * @file    TestStreamlessKernel.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Software Tests for Streamless-Kernel
 */

#include "MatrixProfileTests.hpp"
#include "MockInitialize.hpp"

namespace MatrixProfileTests {

    // wrapper struct to have multiple StreamlessKernel instances
    // with different confugrations (data_t, index_t, n, m)
    // this allows for multiple test (without having to recompile)
    template<typename data_t, typename index_t, int n, int m>
    struct StreamlessKernel: public MatrixProfileKernel<data_t, index_t, n, m> {
        // "negative infinity" used to initialize aggregates
        static constexpr data_t aggregate_init = AggregateInit<data_t>();

        // used to indicate an invalid/undetermined index
        static constexpr index_t index_init = IndexInit<index_t>();

        struct aggregate_t { 
            data_t value; index_t index; 
            aggregate_t() = default;
            aggregate_t(const data_t value, const index_t index)
                : value(value), index(index) {}
            bool operator<(const data_t other) const { return value > other; }
            bool operator>(const aggregate_t other) const { return value > other.value; }
        };

        using ComputePack = typename MatrixProfileKernel<data_t, index_t, n, m>::ComputePack;
        #include "MatrixProfileKernelStreamless.cpp"
    };

    TEST(TestStreamlessKernel, TestSmall8SynM4) {
        StreamlessKernel<double, int, 8, 4> kernel;
        TestMatrixProfileKernel(kernel, "synthetic/small8_syn.txt");
    }

    /*
    TEST(TestStreamlessKernel, TestSmall16SynM4) {
        StreamlessKernel<double, int, 16, 4> kernel;
        TestMatrixProfileKernel(kernel, "synthetic/small16_syn.txt");
    }

    TEST(TestStreamlessKernel, TestSmall128SynM4) {
        StreamlessKernel<double, int, 128, 4> kernel;
        TestMatrixProfileKernel(kernel, "synthetic/small128_syn.txt");
    }

    TEST(TestStreamlessKernel, TestBenchmark1024SynM4) {
        StreamlessKernel<double, int, 1024, 4> kernel;
        TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
    }

    TEST(TestStreamlessKernel, TestBenchmark1024SynM20) {
        StreamlessKernel<double, int, 1024, 20> kernel;
        TestMatrixProfileKernel(kernel, "benchmark/1024.txt");
    }
    */

}