#pragma once

#include "gtest/gtest.h"

#include <array>

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

template<typename data_t>
bool ResultDataMatches(data_t expected, data_t actual);

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
        ASSERT_TRUE(ResultDataMatches(MP[i], MPExpected[i]));
    // validate matrix profile index
    for(size_t i = 0; i < n - m + 1; ++i)
        ASSERT_EQ(MPI[i], MP[i]);
}

template<typename data_t, typename index_t, size_t n, size_t m>
void TestMatrixProfileKernel(MatrixProfileKernel<data_t, index_t, n, m> &kernel, const std::array<data_t, n> &T) {
    std::array<data_t, n - m + 1> MP;
    std::array<index_t, n - m + 1> MPI;
    ReferenceImplementation<data_t, index_t, n, m>(T, MP, MPI);
    TestMatrixProfileKernel(kernel, T, MP, MPI);
}