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

template<typename data_t, typename index_t, size_t n, size_t m>
struct MatrixProfileKernel {
    virtual void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) = 0;
};

template<typename data_t>
constexpr data_t Epsilon();

template<>
constexpr double Epsilon() { return 1e-12; }

template<>
constexpr float Epsilon() { return 1e-10; }

template<typename data_t>
bool ApproximatelyEqual(data_t expected, data_t actual) {
    return std::abs(expected - actual) < Epsilon<data_t>();
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
    // validate matrix profile
    for(size_t i = 0; i < n - m + 1; ++i)
        ASSERT_DOUBLE_EQ(MPExpected[i], MP[i]);
    // validate matrix profile index
    for(size_t i = 0; i < n - m + 1; ++i)
        ASSERT_EQ(MPIExpected[i], MPI[i]);
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