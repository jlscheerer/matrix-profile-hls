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
#include "kernel/TreeReduce.hpp"
#include "MockStream.hpp"
#include "MockInitialize.hpp"
using Mock::stream;

#include "MatrixProfileReference.hpp"


#define aggregate_t_init (aggregate_t){aggregate_init, index_init}
#define sublen (n - m + 1)

namespace MatrixProfileTests {

    template<typename data_t, typename index_t, int w>
    struct MatrixProfileKernel {

        // "negative infinity" used to initialize aggregates
        static constexpr data_t aggregate_init = AggregateInit<data_t>();

        // used to indicate an invalid/undetermined index
        static constexpr index_t index_init = IndexInit<index_t>();

        struct aggregate_t { 
            data_t value; index_t index; 
            aggregate_t() = default;
            aggregate_t(const data_t value, const index_t index)
                : value(value), index(index) {}
            bool operator<(const data_t other) const { return value < other; }
            bool operator>(const aggregate_t other) const { return value > other.value; }
        };

        struct InputDataPack {
            data_t QT, df, dg, inv;
            InputDataPack() = default;
            InputDataPack(data_t value)
                : QT(value), df(value), dg(value), inv(value) {}
        };

        struct OutputDataPack {
            aggregate_t rowAggregate, columnAggregate;
            OutputDataPack() = default;
            OutputDataPack(aggregate_t rowAggregate, aggregate_t columnAggregate)
                : rowAggregate(rowAggregate), columnAggregate(columnAggregate) {}
        };

        static constexpr index_t nColumns = w;

        virtual void MatrixProfileKernelTLF(const index_t n, const index_t m, const index_t iteration, const InputDataPack *columns, const InputDataPack *rows, OutputDataPack *out) = 0;
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
    void PearsonCorrelationToEuclideanDistance(std::array<data_t, n - m + 1> &MP) {
        for (index_t i = 0; i < n - m + 1; ++i)
            MP[i] = std::sqrt(2 * m * (1 - MP[i]));
    }

    template<typename data_t, typename index_t, size_t n, size_t m, typename InputDataPack>
    void PrecomputeStatistics(const std::array<data_t, n> &T, std::array<InputDataPack, n - m + 1> &data) {
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
            data[i].QT = 0; data[i].inv = 0;
            for (index_t k = 0; k < m; ++k) {
                data[i].QT += (T[i + k] - mean) * (T[k] - mu0);
                data[i].inv += (T[i + k] - mean) * (T[i + k] - mean);
            }
            data[i].inv = static_cast<data_t>(1) / std::sqrt(data[i].inv);
        }
    }

    template<typename data_t, typename index_t, size_t n, size_t m, int w>
    void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, w> &kernel, const std::array<data_t, n> &T, 
                                const std::array<data_t, n - m + 1> &MPExpected, const std::array<index_t, n - m + 1> &MPIExpected) {
        using aggregate_t = typename MatrixProfileKernel<data_t, index_t, w>::aggregate_t;
        using InputDataPack = typename MatrixProfileKernel<data_t, index_t, w>::InputDataPack;
        using OutputDataPack = typename MatrixProfileKernel<data_t, index_t, w>::OutputDataPack;
        
        constexpr data_t aggregate_init = MatrixProfileKernel<data_t, index_t, w>::aggregate_init;
        constexpr index_t index_init = MatrixProfileKernel<data_t, index_t, w>::index_init;
        constexpr index_t nColumns = MatrixProfileKernel<data_t, index_t, w>::nColumns;

        Mock::Reset();

        std::array<aggregate_t, n - m + 1> rowAggregates, columnAggregates;
        std::array<InputDataPack, n - m + 1> input;
        PrecomputeStatistics<data_t, index_t, n, m, InputDataPack>(T, input);

        std::array<OutputDataPack, n - m + 1> output;

        constexpr index_t nIterations = (n - m + nColumns) / nColumns;
        for (index_t iteration = 0; iteration < nIterations; ++iteration) {
            const index_t nOffset = iteration * nColumns;
            const index_t nRows = n - m + 1 - nOffset;
            
            kernel.MatrixProfileKernelTLF(n, m, iteration, input.data(), input.data(), output.data());

            // Update Local "Copies" of Aggregates
            for (index_t i = 0; i < nRows; ++i) {
                aggregate_t prevRow = iteration > 0 ? rowAggregates[i] : aggregate_t_init;
                aggregate_t prevCol = iteration > 0 ? columnAggregates[i + nOffset] : aggregate_t_init;

                aggregate_t currRow = output[i].rowAggregate;
                aggregate_t currCol = output[i].columnAggregate;

                rowAggregates[i] = currRow.value > prevRow.value ? currRow : prevRow;
                columnAggregates[i + nOffset] = currCol.value > prevCol.value ? currCol : prevCol;
            }
        }

        std::array<data_t, n - m + 1> MP;
        std::array<index_t, n - m + 1> MPI;

        // merge aggregates at the "very" end
        for (index_t i = 0; i < n - m + 1; ++i) {
            aggregate_t rowAggregate = rowAggregates[i];
            aggregate_t columnAggregate = columnAggregates[i];
        
            // merge the aggregates by taking the maximum
            aggregate_t aggregate = rowAggregate.value > columnAggregate.value ? rowAggregate : columnAggregate;

            // directly convert obtained pearson correlation to euclidean distance
            MP[i] = aggregate.value;
            MPI[i] = aggregate.index;
        }

        // Convert Pearson Correlation to Euclidean Distance
        PearsonCorrelationToEuclideanDistance<data_t, index_t, n, m>(MP);

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

    template<typename data_t, typename index_t, size_t n, size_t m, int w>
    void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, w> &kernel, const std::array<data_t, n> &T) {
        std::array<data_t, n - m + 1> MP;
        std::array<index_t, n - m + 1> MPI;
        Reference::ComputeMatrixProfile<data_t, index_t, n, m>(T, MP, MPI);
        TestMatrixProfileKernel<data_t, index_t, n, m, w>(kernel, T, MP, MPI);
    }

    template<typename data_t, typename index_t, size_t n, size_t m, int w>
    void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, w> &kernel, const std::string &inputFile) {
        std::ifstream input("../data/" + inputFile);
        ASSERT_TRUE(input.is_open());
        std::array<data_t, n> T;
        for(size_t i = 0; i < n; ++i)
            input >> T[i];
        TestMatrixProfileKernel<data_t, index_t, n, m, w>(kernel, T);
    }

}