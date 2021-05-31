#pragma once

#include <cstdlib>
#include <array>

template<typename data_t, typename index_t, size_t n, size_t m>
void ReferenceImplementation(const std::array<data_t, n> &T, std::array<data_t, n - m + 1> &MP, std::array<index_t, n - m + 1> &MPI) {

}