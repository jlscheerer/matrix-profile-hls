/**
 * @file    HostSideComputation.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Host-Side Computation (Pre-Computation & Post-Computation)
 */

#pragma once

#include "Config.hpp"

#include "host/Timer.hpp"
#include "host/BenchmarkProfile.hpp"

#include <array>
#include <numeric>
#include <cmath>

namespace HostSideComputation {

    static void PreComputeStatistics(BenchmarkProfile &profile, std::array<double, n> &T, std::array<InputDataPack, n - m + 1> &data) {
        Timer timer;
        // Calculate the initial mean, then update using moving mean.
        double mean = std::accumulate(T.begin(), T.begin() + m, static_cast<double>(0)); mean /= m;
        double prev_mu, mu0 = mean;

        // Compute Statistics in single pass through the data
        for (index_t i = 0; i < n - m + 1; ++i) {
            prev_mu = mean;
            mean += (i > 0) ? ((T[i + m - 1] - T[i - 1]) / m) : 0;
            data[i].df = (i > 0) ? static_cast<data_t>((T[i + m - 1] - T[i - 1]) / 2)
                            : static_cast<data_t>(0);
            data[i].dg = (i > 0) ? static_cast<data_t>(((T[i + m - 1] - mean) + (T[i - 1] - prev_mu))) 
                            : static_cast<data_t>(0);

	        double qt = 0, inv = 0;
            for (index_t k = 0; k < m; ++k) {
                qt += (T[i + k] - mean) * (T[k] - mu0);
                inv += (T[i + k] - mean) * (T[i + k] - mean);
            }
            data[i].QT = qt;
            data[i].inv = (1 / std::sqrt(inv));
        }
        const auto time = timer.Elapsed();
        profile.Push("1. Host-Side [Pre-Computation]", time);
    }

    static double PearsonCorrelationToEuclideanDistance(const index_t n, const index_t m, const double P) {
        return std::sqrt(2 * m * (1 - P));
    }

    static void PostComputeAggregates(BenchmarkProfile &profile, const std::array<aggregate_t, n - m + 1> &rowAggregates,
                               const std::array<aggregate_t, n - m + 1> &columnAggregates, std::array<double, n - m + 1> &MP,
                               std::array<index_t, n - m + 1> &MPI) {
        Timer timer;
        // merge aggregates at the "very" end
        for (index_t i = 0; i < n - m + 1; ++i) {
            aggregate_t rowAggregate = rowAggregates[i];
            aggregate_t columnAggregate = columnAggregates[i];
        
            // merge the aggregates by taking the maximum
            aggregate_t aggregate = rowAggregate.value > columnAggregate.value ? rowAggregate : columnAggregate;

            // directly convert obtained pearson correlation to euclidean distance
            MP[i] = PearsonCorrelationToEuclideanDistance(n, m, aggregate.value);
            MPI[i] = aggregate.index;
        }
        const auto time = timer.Elapsed();
        profile.Push("4. Host-Side [Post-Computation]", time);
    }

}

