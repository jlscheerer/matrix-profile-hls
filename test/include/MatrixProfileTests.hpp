/**
 * @file    MatrixProfileTests.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Software Test Utility functions for Matrix Profiles Computation
 */

#pragma once

#include "gtest/gtest.h"

#include <fstream>
#include <array>
#include <string>

#include <cstdlib>
#include <cmath>

#define TEST_MOCK_SW
#include "MockStream.hpp"
using Mock::stream;

#include "MatrixProfileReference.hpp"

#define aggregate_t_init (aggregate_t){aggregate_init, index_init}
#define sublen (n - m + 1)

namespace MatrixProfileTests {

    template<typename data_t, typename index_t, size_t n, size_t m>
    struct MatrixProfileKernel {
        virtual void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) = 0;
    };

    template<typename data_t>
    constexpr data_t Epsilon();

    template<>
    constexpr double Epsilon() { return 1e-8; }

    template<typename data_t>
    bool ApproximatelyEqual(data_t expected, data_t actual) {
        return std::abs(expected - actual) < Epsilon<data_t>();
    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    data_t MeanOfSubsequence(const std::array<data_t, n> &T, index_t i) {
        data_t mu = 0;
        for (index_t k = 0; k < m; ++k)
            mu += T[i + k];
        return mu / m;
    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    data_t StdOfSubsequence(const std::array<data_t, n> &T, index_t i, data_t mui) {
        data_t std = 0;
        for (index_t k = 0; k < m; ++k)
            std += T[i + k] * T[i + k];
        std /= m;
        return std::sqrt(std - mui * mui);
    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    data_t EuclideanDistance(const std::array<data_t, n> &T, index_t i, index_t j) {
        data_t mui{MeanOfSubsequence<data_t, index_t, n, m>(T, i)}, muj{MeanOfSubsequence<data_t, index_t, n, m>(T, j)};
        data_t stdi{StdOfSubsequence<data_t, index_t, n, m>(T, i, mui)}, stdj{StdOfSubsequence<data_t, index_t, n, m>(T, j, muj)};
        data_t distance = 0;
        for (index_t k = 0; k < m; ++k) {
            data_t xi = (T[i + k] - mui) / stdi;
            data_t yi = (T[j + k] - muj) / stdi;
            distance += (xi - yi) * (xi - yi);
        }
        return std::sqrt(distance);
    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, n, m> &kernel, const std::array<data_t, n> &T, 
                                const std::array<data_t, n - m + 1> &MPExpected, const std::array<index_t, n - m + 1> &MPIExpected) {
        Mock::Reset();
        std::array<data_t, n - m + 1> MP;
        std::array<index_t, n - m + 1> MPI;
        kernel.MatrixProfileKernelTLF(T.data(), MP.data(), MPI.data());
        // check that in fact all streams are empty
        ASSERT_EQ(Mock::all_streams_empty, true);
        // check that we never read from an empty stream
        ASSERT_EQ(Mock::read_from_empty_stream, false);
        // validate matrix profile / matrix profile index
        for(size_t i = 0; i < n - m + 1; ++i) {
            ASSERT_LE(std::abs(MPExpected[i] - MP[i]), Epsilon<data_t>());
            if(MPIExpected[i] == MPI[i])
                continue;
            // MPI can be different iff actual difference is still within threshold
            data_t d1 = EuclideanDistance<data_t, index_t, n, m>(T, i, MPI[i]);
            data_t d2 = EuclideanDistance<data_t, index_t, n, m>(T, i, MPIExpected[i]);
            ASSERT_FLOAT_EQ(d1, d2);
        }
    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, n, m> &kernel, const std::array<data_t, n> &T) {
        std::array<data_t, n - m + 1> MP;
        std::array<index_t, n - m + 1> MPI;
        Reference::ComputeMatrixProfile<data_t, index_t, n, m>(T, MP, MPI);
        TestMatrixProfileKernel(kernel, T, MP, MPI);
    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, n, m> &kernel, const std::string &inputFile) {
        std::ifstream input("../data/" + inputFile);
        ASSERT_TRUE(input.is_open());
        std::array<data_t, n> T;
        for(size_t i = 0; i < n; ++i)
            input >> T[i];
        TestMatrixProfileKernel(kernel, T);
    }

}