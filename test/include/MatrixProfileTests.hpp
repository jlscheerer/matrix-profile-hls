#pragma once

#include "gtest/gtest.h"

#include <array>

#include <limits>
#include <cstdlib>
#include <cmath>

#define TEST_MOCK_SW
#include "MockStream.hpp"
using mock::stream;

#include "MatrixProfileReference.hpp"

template<typename data_t, typename index_t, size_t n, size_t m>
struct MatrixProfileKernel {
    virtual void MatrixProfileKernelTLF(const data_t *T, data_t *MP, index_t *MPI) = 0;
};

// TODO: Need to abort calling function
template<typename data_t>
void AssertApproximatelyEqual(data_t MPExpected, data_t MPActual);

template<>
void AssertApproximatelyEqual(double MPExpected, double MPActual) { ASSERT_DOUBLE_EQ(MPExpected, MPActual); }

template<>
void AssertApproximatelyEqual(float MPExpected, float MPActual) { ASSERT_FLOAT_EQ(MPExpected, MPActual); }

template<typename data_t, typename index_t, size_t n, size_t m>
void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, n, m> &kernel, const std::array<data_t, n> &T, 
                             const std::array<data_t, n - m + 1> &MPExpected, const std::array<index_t, n - m + 1> &MPIExpected) {
    mock::reset();
    std::array<data_t, n - m + 1> MP;
    std::array<index_t, n - m + 1> MPI;
    kernel.MatrixProfileKernelTLF(T.data(), MP.data(), MPI.data());
    // check that in fact all streams are empty
    ASSERT_EQ(mock::all_streams_empty, true);
    // check that we never read from an empty stream
    ASSERT_EQ(mock::read_from_empty_stream, false);
    // validate matrix profile
    for(size_t i = 0; i < n - m + 1; ++i)
        AssertApproximatelyEqual(MPExpected[i], MP[i]);
    // validate matrix profile index
    for(size_t i = 0; i < n - m + 1; ++i)
        ASSERT_EQ(MPIExpected[i], MPI[i]);
}

template<typename data_t, typename index_t, size_t n, size_t m>
void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, n, m> &kernel, const std::array<data_t, n> &T) {
    std::array<data_t, n - m + 1> MP;
    std::array<index_t, n - m + 1> MPI;
    ReferenceImplementation<data_t, index_t, n, m>(T, MP, MPI);
    TestMatrixProfileKernel(kernel, T, MP, MPI);
}