#pragma once

#include <fstream>
#include <array>
#include <string>

#include <host/Logger.hpp>

using Logger::Log;
using Logger::LogLevel;

template<typename T, size_t n>
bool ReadBinaryFile(const std::string &fileName, std::array<T, n> &block){
    std::ifstream bin(fileName, std::ios::in | std::ios::binary | std::ios::ate);
    
    if(!bin.is_open()){
        Log<LogLevel::Error>("Failed to open", fileName);
        return false;
    }

    // Determine the size of the file in bytes
    bin.seekg(0, bin.end);
    const auto fileSize = bin.tellg();
    bin.seekg(0, bin.beg);

    if(fileSize != n * sizeof(T)){
        Log<LogLevel::Error>(fileName, "contains unexpected number of elements!");
        Log<LogLevel::Error>("Expected", n, "element(s) [i.e.", n * sizeof(T), "bytes]; but file contains", fileSize, "bytes");
        return false;
    }

    // Read entire contents of the file
    bin.read((char*)block.data(), fileSize);
    bin.close();
    return true;
}

template<typename T, size_t n>
bool WriteBinaryFile(const std::string &fileName, std::array<T, n> &block){
    std::ofstream bin(fileName, std::ios::out | std::ios::binary | std::ios::ate);
    
    if(!bin.is_open()){
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
