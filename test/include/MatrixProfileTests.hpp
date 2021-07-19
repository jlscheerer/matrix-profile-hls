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
#include <numeric>

#define TEST_MOCK_SW
#include "MockStream.hpp"
using Mock::stream;

#include "MatrixProfileReference.hpp"

#define aggregate_t_init (aggregate_t){aggregate_init, index_init}
#define sublen (n - m + 1)

namespace MatrixProfileTests {

    template<typename data_t, typename index_t, size_t n, size_t m>
    struct MatrixProfileKernel {
        struct ComputePack { data_t df, dg, inv; };
        virtual void MatrixProfileKernelTLF(const data_t *QT, const ComputePack *data, data_t *MP, index_t *MPI) = 0;
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
    void PearsonCorrelationToEuclideanDistance(const std::array<data_t, n - m + 1> &PearsonCorrelations, std::array<data_t, n - m + 1> &EuclideanDistance) {
        for (index_t i = 0; i < n - m + 1; ++i)
            EuclideanDistance[i] = std::sqrt(2 * m * (1 - PearsonCorrelations[i]));
    }

    template<typename data_t, typename index_t, size_t n, size_t m, typename ComputePack>
    void PrecomputeStatistics(const std::array<data_t, n> &T, std::array<data_t, n - m + 1> &QT, std::array<ComputePack, n - m + 1> &data) {
        // Calculate the initial mean, then update using moving mean.
        data_t mean = std::accumulate(T.begin(), T.begin() + m, static_cast<data_t>(0)); mean /= m;
        data_t prev_mu, mu0 = mean;

        // Compute Statistics in single pass through the data
        for (index_t i = 0; i < n - m + 1; ++i) {
            prev_mu = mean;
            mean += (i > 0) ? ((T[i + m - 1] - T[i - 1]) / m) 
                            : 0;
            data[i].df = (i > 0) ? (T[i + m - 1] - T[i - 1]) / 2 
                                    : static_cast<data_t>(0);
            data[i].dg = (i > 0) ? ((T[i + m - 1] - mean) + (T[i - 1] - prev_mu)) 
                                    : static_cast<data_t>(0);
            QT[i] = 0; data[i].inv = 0;
            for (index_t k = 0; k < m; ++k) {
                QT[i]  += (T[i + k] - mean) * (T[k] - mu0);
                data[i].inv += (T[i + k] - mean) * (T[i + k] - mean);
            }
            data[i].inv = static_cast<data_t>(1) / std::sqrt(data[i].inv);
        }
    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, n, m> &kernel, const std::array<data_t, n> &T, 
                                const std::array<data_t, n - m + 1> &MPExpected, const std::array<index_t, n - m + 1> &MPIExpected) {
        Mock::Reset();
        std::array<data_t, n - m + 1> P, MP;
        std::array<index_t, n - m + 1> MPI;

        using ComputePack = typename MatrixProfileKernel<data_t, index_t, n, m>::ComputePack;

        std::array<data_t, n - m + 1> QT; std::array<ComputePack, n - m + 1> data;
        PrecomputeStatistics<data_t, index_t, n, m, ComputePack>(T, QT, data);
        kernel.MatrixProfileKernelTLF(QT.data(), data.data(), P.data(), MPI.data());

        // Convert Pearson Correlation to Euclidean Distance
        PearsonCorrelationToEuclideanDistance<data_t, index_t, n, m>(P, MP);

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