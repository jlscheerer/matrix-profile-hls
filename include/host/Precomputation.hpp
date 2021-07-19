#pragma once

#include "Config.hpp"

#include <array>
#include <numeric>

namespace MatrixProfileUtil {

    void PrecomputeStatistics(std::array<data_t, n> &T, std::array<data_t, n - m + 1> &QT, 
                              std::array<data_t, n - m + 1> &df, std::array<data_t, n - m + 1> &dg, 
                              std::array<data_t, n - m + 1> &inv) {
        // Calculate the initial mean, then update using moving mean.
        data_t mean = std::accumulate(T.begin(), T.begin() + m, static_cast<data_t>(0)); mean /= m;
        data_t prev_mu, mu0 = mean;

        // Compute Statistics in single pass through the data
        for (index_t i = 0; i < n - m + 1; ++i) {
            prev_mu = mean;
            mean += (i > 0) ? ((T[i + m - 1] - T[i - 1]) / m) 
                            : 0;
            df[i] = (i > 0) ? (T[i + m - 1] - T[i - 1]) / 2 
                            : static_cast<data_t>(0);
            dg[i] = (i > 0) ? ((T[i + m - 1] - mean) + (T[i - 1] - prev_mu)) 
                            : static_cast<data_t>(0);
            QT[i] = 0; inv[i] = 0;
            for (index_t k = 0; k < m; ++k) {
                QT[i]  += (T[i + k] - mean) * (T[k] - mu0);
                inv[i] += (T[i + k] - mean) * (T[i + k] - mean);
            }
            inv[i] = static_cast<data_t>(1) / inv[i];
        }
    }

}