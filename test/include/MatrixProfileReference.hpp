/**
 * @file    MatrixProfileReference.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Reference Implementation of the SCAMP Algorithm
 */

#pragma once

#include <cstdlib>
#include <cmath>
#include <array>
#include <iostream>

namespace Reference {

    namespace Internal {

        template<typename data_t>
        constexpr data_t AggregateInit();

        template<>
        constexpr double AggregateInit() { return -1e12; }

        template<typename index_t>
        constexpr index_t IndexInit();

        template<>
        constexpr int IndexInit() { return -1; }

        template<typename data_t, typename index_t, index_t n, index_t m>
        std::array<data_t, n - m + 1> MovMean(const std::array<data_t, n> &T) {
            std::array<data_t, n - m + 1> mu{0};
            // calculate the mean for the first m points directly
            for (index_t i = 0; i < m; ++i)
                mu[0] += T[i];
            mu[0] /= m;
            // calculate the rest of the mean iteratively
            for (index_t i = 1; i < n - m + 1; ++i)
                mu[i] = mu[i - 1] + (T[i + m - 1] - T[i - 1]) / m;
            return mu;
        }

        template<typename data_t, typename index_t, index_t n, index_t m>
        std::array<data_t, n - m + 1> ComputeDf(const std::array<data_t, n> &T) {
            std::array<data_t, n - m + 1> df{0};
            for (index_t i = 1; i  < n - m + 1; ++i)
                df[i] = (T[i + m - 1] - T[i - 1]) / 2;
            return df;
        }

        template<typename data_t, typename index_t, index_t n, index_t m>
        std::array<data_t, n - m + 1> ComputeDg(const std::array<data_t, n> &T, const std::array<data_t, n - m + 1> &mu) {
            std::array<data_t, n - m + 1> dg{0};
            for (index_t i = 1; i < n - m + 1; ++i)
                dg[i] = (T[i + m - 1] - mu[i]) + (T[i - 1] - mu[i - 1]);
            return dg;
        }

        template<typename data_t, typename index_t, index_t n, index_t m>
        std::array<data_t, n - m + 1> ComputeInv(const std::array<data_t, n> &T, const std::array<data_t, n - m + 1> &mu) {
            std::array<data_t, n - m + 1> inv;
            for (index_t i = 0; i < n - m + 1; ++i) {
                data_t sum = 0;
                for (index_t k = 0; k < m; ++k)
                    sum += (T[i + k] - mu[i]) * (T[i + k] - mu[i]);
                inv[i] = 1 / sqrt(sum);
            }
            return inv;
        }

        // Assumption: column >= row
        template<typename data_t, typename index_t, index_t n, index_t m>
        inline bool ExclusionZone(index_t row, index_t column) {
            // exclusionZone <==> row - m/4 <= column <= row + m/4
            //               <==> column <= row + m/4 [(row <= column, m > 0) ==> row - m/4 <= column]
            //               <==> row + (column - row) <= row + m/4
            //               <==> (column - row) <= m/4
            return (column - row) < m/4;
        }

        template<typename data_t, typename index_t, index_t n, index_t m>
        inline void UpdateMatrixProfile(index_t i, index_t j, data_t PearsonCorrelation, std::array<data_t, n - m + 1> &MP, std::array<index_t, n - m + 1> &MPI) {
            if(PearsonCorrelation > MP[i]) {
                MP[i] = PearsonCorrelation; MPI[i] = j;
            }
        }

    }

    template<typename data_t, typename index_t, index_t n, index_t m>
    void ComputeMatrixProfile(const std::array<data_t, n> &T, std::array<data_t, n - m + 1> &MP, std::array<index_t, n - m + 1> &MPI) {
        using namespace Internal;
        // Precompute Required Statistics
        std::array<data_t, n - m + 1> mu{MovMean<data_t, index_t, n, m>(T)};
        std::array<data_t, n - m + 1> df{ComputeDf<data_t, index_t, n, m>(T)};
        std::array<data_t, n - m + 1> dg{ComputeDg<data_t, index_t, n, m>(T, mu)};
        std::array<data_t, n - m + 1> inv{ComputeInv<data_t, index_t, n, m>(T, mu)};

        // Initialize Results
        std::fill(MP.begin(), MP.end(), AggregateInit<data_t>());
        std::fill(MPI.begin(), MPI.end(), IndexInit<index_t>());

        // Compute the first Row of the Matrix
        std::array<data_t, n - m + 1> QT;
        data_t PearsonCorrelation;
        for (index_t i = 0; i < n - m + 1; ++i) {
            data_t sum = 0;
            for (index_t k = 0; k < m; ++k)
                sum += (T[i + k] - mu[i]) * (T[k] - mu[0]);
            QT[i] = sum;
            PearsonCorrelation = QT[i] * inv[0] * inv[i];
            if (!ExclusionZone<data_t, index_t, n, m>(0, i)) {
                // Update Row-Wise Minimum
                UpdateMatrixProfile<data_t, index_t, n, m>(0, i, PearsonCorrelation, MP, MPI);
                // Update Column-Wise Minimum
                UpdateMatrixProfile<data_t, index_t, n, m>(i, 0, PearsonCorrelation, MP, MPI);
            }
        }

        for (index_t row = 1; row < n - m + 1; ++row) {
            for (index_t k = 0; k < n - m + 1 - row; ++k) {
                QT[k] += df[row] * dg[k + row] + df[k + row] * dg[row];
                PearsonCorrelation = QT[k] * inv[row] * inv[k + row];
                if (!ExclusionZone<data_t, index_t, n, m>(row, k + row)) {
                    // Update Row-Wise Minimum
                    UpdateMatrixProfile<data_t, index_t, n, m>(row, k + row, PearsonCorrelation, MP, MPI);
                    // Update Column-Wise Minimum
                    UpdateMatrixProfile<data_t, index_t, n, m>(k + row, row, PearsonCorrelation, MP, MPI);
                }
            }
        }

        // Convert from Pearson Correlation to Euclidean Distance
        for (index_t i = 0; i < n - m + 1; ++i)
            MP[i] = sqrt(2 * m * (1 - MP[i]));
    }

}