/**
 * @file    FileIO.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Utility for reading and writing Binary Files
 */

#pragma once

#include <fstream>
#include <array>
#include <string>

#include "ap_fixed.h"

#include "Config.hpp"

#include <host/Logger.hpp>

using Logger::Log;
using Logger::LogLevel;

namespace FileIO {

    template<typename T, size_t n>
    static bool ReadBinaryFile(const std::string &fileName, std::array<T, n> &block) {
        std::ifstream bin(fileName, std::ios::in | std::ios::binary | std::ios::ate);
        
        if(!bin.is_open()){
            Log<LogLevel::Error>("Failed to open", fileName);
            return false;
        }

        // Determine the size of the file in bytes
        bin.seekg(0, bin.end);
        const auto fileSize = bin.tellg();
        bin.seekg(0, bin.beg);

        if (fileSize != n * sizeof(T)) {
            Log<LogLevel::Error>(fileName, "contains unexpected number of elements!");
            Log<LogLevel::Error>("Expected", n, "element(s) [i.e.", n * sizeof(T), "bytes]; but file contains", fileSize, "bytes");
            return false;
        }

        // Read entire contents of the file
        bin.read((char*)block.data(), fileSize);
        bin.close();
        return true;
    }

    template<int W, int I>
    static constexpr double ap_min_value() {
        // Do not use the absolute min value, causes rounding errors!
        // Actual min value: -((1LL << (I-1))) - ((double)(1LL << (W-I))-1) / (1LL << (W-I));
        // Use min integral value instead
        return -((1LL << (I-1)));
    }

    template<int W, int I>
    static constexpr double ap_max_value() {
        // Do not use the absolute min value, causes rounding errors!
        // Actual min value: ((1LL << (I-1))-1) + ((double)(1LL << (W-I))-1) / (1LL << (W-I));
        // Use min integral value instead
        return ((1LL << (I-1))-1);
    }

    template<int W, int I, size_t n>
    static bool ReadBinaryFile(const std::string &fileName, std::array<ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM>, n> &block) {
        Log<LogLevel::Verbose>("Reading", fileName, "as type double and casting to ap_fixed");
        std::ifstream bin(fileName, std::ios::in | std::ios::binary | std::ios::ate);
        
        if(!bin.is_open()){
            Log<LogLevel::Error>("Failed to open", fileName);
            return false;
        }

        // Determine the size of the file in bytes
        bin.seekg(0, bin.end);
        const auto fileSize = bin.tellg();
        bin.seekg(0, bin.beg);

        if (fileSize != n * sizeof(double)) {
            Log<LogLevel::Error>(fileName, "contains unexpected number of elements!");
            Log<LogLevel::Error>("Expected", n, "element(s) [i.e.", n * sizeof(double), "bytes]; but file contains", fileSize, "bytes");
            return false;
        }

        // Read Input File as Double and Perform Checks
        double value;
        for (size_t i = 0; i < n; ++i) {
            bin.read((char*)&value, sizeof(double));
            #if _io_chk_ap_range
                // check if input value is in range
                if (value < ap_min_value<W, I>() || value > ap_max_value<W, I>()) {
                    Log<LogLevel::Error>(fileName, "contains value", value, "not contained in \"safe-range\"! Expected value between", ap_min_value<W, I>(), "and", ap_max_value<W, I>(), "");
                    return false;
                }
            #endif
            block[i] = static_cast<ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM>>(value);
        }

        bin.close();
        return true;
    }

    template<typename T, size_t n>
    static bool WriteBinaryFile(const std::string &fileName, std::array<T, n> &block) {
        std::ofstream bin(fileName, std::ios::out | std::ios::binary | std::ios::ate);
        
        if (!bin.is_open()) {
            Log<LogLevel::Error>("Failed to open", fileName, "for writing");
            return false;
        }

        // Determine the required file size in number of bytes
        const auto fileSize = n * sizeof(T);

        // Write entire content to the file
        bin.write((char*)block.data(), fileSize);
        bin.close();
        return true;
    }

    template<int W, int I, size_t n>
    static bool WriteBinaryFile(const std::string &fileName, std::array<ap_fixed<W, I, AP_RND_ZERO, AP_WRAP_SM>, n> &block) {
        Log<LogLevel::Verbose>("Writing", fileName, "as type double and casting from ap_fixed");
        std::ofstream bin(fileName, std::ios::out | std::ios::binary | std::ios::ate);
        
        if (!bin.is_open()) {
            Log<LogLevel::Error>("Failed to open", fileName, "for writing");
            return false;
        }

        // Write Output File as Double(s)
        double value;
        for (size_t i = 0; i < n; ++i) {
            value = static_cast<double>(block[i]);
            bin.write((char*)&value, sizeof(double));
        }

        bin.close();
        return true;
    }
}