#pragma once

#include <cstdlib>
#include <cmath>
#include <array>
#include <iostream>

namespace Reference {

    namespace Internal {

        template<typename data_t, size_t n, size_t m>
        std::array<data_t, n - m + 1> MovMean(const std::array<data_t, n> &T) {
            std::array<data_t, n - m + 1> mu{0};
            // calculate the mean for the first m points directly
            for (size_t i = 0; i < m; ++i)
                mu[0] += T[i];
            mu[0] /= m;
            // calculate the rest of the mean iteratively
            for (size_t i = 1; i < n - m + 1; ++i)
                mu[i] = mu[i - 1] + (T[i + m - 1] - T[i - 1]) / m;
            return mu;
        }

        template<typename data_t, size_t n, size_t m>
        std::array<data_t, n - m + 1> ComputeDf(const std::array<data_t, n> &T) {
            std::array<data_t, n - m + 1> df{0};
            for (size_t i = 1; i  < n - m + 1; ++i)
                df[i] = (T[i + m - 1] - T[i - 1]) / 2;
            return df;
        }

        template<typename data_t, size_t n, size_t m>
        std::array<data_t, n - m + 1> ComputeDg(const std::array<data_t, n> &T, const std::array<data_t, n - m + 1> &mu) {
            std::array<data_t, n - m + 1> dg{0};
            for (size_t i = 1; i < n - m + 1; ++i)
                dg[i] = (T[i + m - 1] - mu[i]) + (T[i - 1] - mu[i - 1]);
            return dg;
        }

        template<typename data_t, size_t n, size_t m>
        std::array<data_t, n - m + 1> ComputeInv(const std::array<data_t, n> &T, const std::array<data_t, n - m + 1> &mu) {
            std::array<data_t, n - m + 1> inv;
            for (size_t i = 0; i < n - m + 1; ++i) {
                data_t sum = 0;
                for (size_t k = 0; k < m; ++k)
                    sum += (T[i + k] - mu[i]) * (T[i + k] - mu[i]);
                inv[i] = 1 / std::sqrt(sum);
            }
            return inv;
        }

        // Assumption: column >= row
        template<typename data_t, size_t n, size_t m>
        inline bool ExclusionZone(size_t row, size_t column) {
            // exclusionZone <==> row - m/4 <= column <= row + m/4
            //               <==> column <= row + m/4 [(row <= column, m > 0) ==> row - m/4 <= column]
            //               <==> row + (column - row) <= row + m/4
            //               <==> (column - row) <= m/4
            return (column - row) <= m/4;
        }

        template<typename data_t, typename index_t, size_t n, size_t m>
        inline void UpdateMatrixProfile(size_t i, size_t j, data_t PearsonCorrelation, std::array<data_t, n - m + 1> &MP, std::array<index_t, n - m + 1> &MPI) {
            if(PearsonCorrelation > MP[i]) {
                MP[i] = PearsonCorrelation; MPI[i] = j;
            }
        }

    }

    template<typename data_t, typename index_t, size_t n, size_t m>
    void ComputeMatrixProfile(const std::array<data_t, n> &T, std::array<data_t, n - m + 1> &MP, std::array<index_t, n - m + 1> &MPI) {
        using namespace Internal;
        // Precompute Required Statistics
        std::array<data_t, n - m + 1> mu{MovMean<data_t, n, m>(T)};
        std::array<data_t, n - m + 1> df{ComputeDf<data_t, n, m>(T)};
        std::array<data_t, n - m + 1> dg{ComputeDg<data_t, n, m>(T, mu)};
        std::array<data_t, n - m + 1> inv{ComputeInv<data_t, n, m>(T, mu)};

        // Initialize Results
        // TODO: Use Type appropriate initializer
        std::fill(MP.begin(), MP.end(), -1e12);
        std::fill(MPI.begin(), MPI.end(), -1);

        // Compute the first Row of the Matrix
        std::array<data_t, n - m + 1> QT;
        data_t PearsonCorrelation;
        for (size_t i = 0; i < n - m + 1; ++i) {
            data_t sum = 0;
            for (size_t k = 0; k < m; ++k)
                sum += (T[i + k] - mu[i]) * (T[k] - mu[0]);
            QT[i] = sum;
            PearsonCorrelation = QT[i] * inv[0] * inv[i];
            if (!ExclusionZone<data_t, n, m>(0, i)) {
                // Update Row-Wise Minimum
                UpdateMatrixProfile<data_t, index_t, n, m>(0, i, PearsonCorrelation, MP, MPI);
                // Update Column-Wise Minimum
                UpdateMatrixProfile<data_t, index_t, n, m>(i, 0, PearsonCorrelation, MP, MPI);
            }
        }

        for (size_t row = 1; row < n - m + 1; ++row) {
            for (size_t k = 0; k < n - m + 1 - row; ++k) {
                QT[k] += df[row] * dg[k + row] + df[k + row] * dg[row];
                PearsonCorrelation = QT[k] * inv[row] * inv[k + row];
                if (!ExclusionZone<data_t, n, m>(row, k + row)) {
                    // Update Row-Wise Minimum
                    UpdateMatrixProfile<data_t, index_t, n, m>(row, k + row, PearsonCorrelation, MP, MPI);
                    // Update Column-Wise Minimum
                    UpdateMatrixProfile<data_t, index_t, n, m>(k + row, row, PearsonCorrelation, MP, MPI);
                }
            }
        }

        // Convert from Pearson Correlation to Euclidean Distance
        for (size_t i = 0; i < n - m + 1; ++i)
            MP[i] = std::sqrt(2 * m * (1 - MP[i]));
    }

}