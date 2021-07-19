#pragma once

#include "Config.hpp"

#include <array>
#include <numeric>
#include <cmath>

namespace HostSideComputation {

    void PrecomputeStatistics(std::array<double, n> &T, std::array<data_t, n - m + 1> &QT, std::array<ComputePack, n - m + 1> &data) {
        // Calculate the initial mean, then update using moving mean.
        double mean = std::accumulate(T.begin(), T.begin() + m, static_cast<data_t>(0)); mean /= m;
        double prev_mu, mu0 = mean;

        // Compute Statistics in single pass through the data
        for (index_t i = 0; i < n - m + 1; ++i) {
            prev_mu = mean;
            mean += (i > 0) ? ((T[i + m - 1] - T[i - 1]) / m) 
                            : 0;
            data[i].df = (i > 0) ? (T[i + m - 1] - T[i - 1]) / 2 
                            : static_cast<data_t>(0);
            data[i].dg = (i > 0) ? ((T[i + m - 1] - mean) + (T[i - 1] - prev_mu)) 
                            : static_cast<data_t>(0);
            double qt = 0, inv = 0;
            for (index_t k = 0; k < m; ++k) {
                qt  += (T[i + k] - mean) * (T[k] - mu0);
                inv += (T[i + k] - mean) * (T[i + k] - mean);
            }
            QT[i] = qt;
            data[i].inv = (1 / std::sqrt(inv));
        }
    }

    void PearsonCorrelationToEuclideanDistance(std::array<data_t, n - m + 1> &PearsonCorrelation, std::array<double, n - m + 1> &EuclideanDistance) {
        for (index_t i = 0; i < n - m + 1; ++i)
            EuclideanDistance[i] = std::sqrt(2 * m * (1 - PearsonCorrelation[i]));
    }

}