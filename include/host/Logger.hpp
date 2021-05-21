/**
 * @file    Logger.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Logging Functionality
 */

#pragma once
#include <iostream>

namespace Logger {

    static bool Verbose = false;
    enum class LogLevel { Info, Verbose, Warning, Error, Debug };

    namespace Internal {
        template <typename T>
        void Log(const T &m){ std::cout << m << std::endl; }

        template <typename T, typename... Args>
        void Log(const T &m, Args const &... r){ std::cout << m << " "; Log(r...); }

        template <typename T>
        void LogError(const T &m){ std::cerr << m << std::endl; }

        template <typename T, typename... Args>
        void LogError(const T &m, Args const &... r){ std::cerr << m << " "; Log(r...); }
    }

    template <LogLevel L, typename T, typename... Args>
    void Log(const T &m, Args const &... r){
        switch(L){
            case LogLevel::Info:
                std::cout << "[INFO] ";
                Internal::Log(m, r...);
                break;
            case LogLevel::Verbose:
                if(Verbose){
                    std::cout << "[INFO/V] ";
                    Internal::Log(m, r...);
                }
                break;
            case LogLevel::Warning:
                std::cout << "[WARN] ";
                Internal::Log(m, r...);
                break;
            case LogLevel::Error:
                std::cerr << "[ERROR] ";
                Internal::LogError(m, r...);
                break;
            case LogLevel::Debug:
                #ifdef DEBUG
                    std::cout << "[DEBUG] ";
                    Internal::Log(m, r...);
                #endif
                break;
        }
    }

}