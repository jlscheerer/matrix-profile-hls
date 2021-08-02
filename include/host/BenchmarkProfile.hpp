/**
 * @file    BenchmarkProfile.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Utilities to Benchmark Computation Times
 */

#pragma once

#include "host/Timer.hpp"

#include <iostream>
#include <chrono>

#include <vector>
#include <string>
#include <map>

struct BenchmarkProfile {

    void Push(const std::string &category, const std::string &name, std::chrono::nanoseconds time) {
        profile[category].emplace_back(name, time);
        times[category] += time;
        totalTime += time;
    }

    void Push(const std::string &category, std::chrono::nanoseconds time) {
        Push(category, std::string(), time);
    }

    void Report() {
        std::cout << std::fixed;
        std::cout.precision(2);

        std::cout << "Generating Performance Report..." << std::endl;
        std::cout << std::endl;
        for (const auto &category: profile) {
            double percentage = static_cast<double>(times[category.first].count()) / totalTime.count() * 100.0;
            std::cout << " " << category.first << ": " << times[category.first] << " \e[30m[" << percentage << "%]\e[0m" << std::endl;
            for (const auto &entry: category.second) {
                if(entry.first.empty()) continue;
                double localPercentage = static_cast<double>(entry.second.count()) / times[category.first].count() * 100.0;
                double globalPercentage = static_cast<double>(entry.second.count()) / totalTime.count() * 100.0;
                std::cout << "     - " << entry.first << ": " << entry.second << " \e[30m[" << localPercentage << "%|" << globalPercentage
                 << "%]\e[0m" << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "\e[1mTotal Execution Time: "<< totalTime << "\e[0m" << std::endl;

        std::resetiosflags(std::ios::showbase);
    }

private:
    std::map<std::string, std::vector<std::pair<std::string, std::chrono::nanoseconds>>> profile;
    std::map<std::string, std::chrono::nanoseconds> times;
    std::chrono::nanoseconds totalTime;
};
