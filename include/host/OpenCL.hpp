#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <string>
#include<vector>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <CL/cl2.hpp>

#include <host/logger.hpp>

#include <optional.hpp>

using tl::optional;
using Logger::Log;
using Logger::LogLevel;

optional<cl::Device> FindDevice(){
    // Traverse all available Platforms to find Xilinx Platform and targeted Device
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl_int errorCode;

    errorCode = cl::Platform::get(&platforms);
    if(errorCode != CL_SUCCESS){
        Log<LogLevel::Error>("Failed to retrieve OpenCL platforms");
        return optional<cl::Device>{};
    }
    for(auto &platform: platforms){
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&errorCode);
        if(errorCode != CL_SUCCESS){
            Log<LogLevel::Error>("Failed to retrieve platform vendor name");
            return optional<cl::Device>{};
        }
        if(platformName == "Xilinx"){
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if(devices.size())
                return tl::make_optional(devices[0]);
        }
    }
    Log<LogLevel::Error>("Failed to find Xilinx-Platform or no available devices found");
    return optional<cl::Device>{};
}

optional<cl::Kernel> MakeKernel(const cl::Context &context, const cl::Device &device, const std::string &xclbin, const std::string &kernelName){
    // Load the xclbin File
    Log<LogLevel::Info>("Loading:", xclbin);
    std::ifstream bin(xclbin, std::ios::in | std::ios::binary | std::ios::ate);

    if(!bin.is_open()){
        Log<LogLevel::Error>("Failed to open .xclbin file", xclbin);
        return optional<cl::Kernel>{};
    }

    // Determine the size of the file in bytes
    bin.seekg(0, bin.end);
    const auto fileSize = bin.tellg();
    bin.seekg(0, bin.beg);

    char *binary = new char[fileSize];
    bin.read(binary, fileSize);

    // The constructor of cl::Program accepts a vector of binaries, so stick the
    // binary into a single-element outer vector.
    cl::Program::Binaries binaries;
    binaries.emplace_back(binary, fileSize);

    std::vector<int> binaryStatus(1);
    cl_int errorCode;

    std::vector<cl::Device> devices;
    devices.emplace_back(device);

    // Create Program from Binary File
    cl::Program program(context, devices, binaries, &binaryStatus, &errorCode);

    if(binaryStatus[0] != CL_SUCCESS){
        Log<LogLevel::Error>("Failed to create OpenCL program from binary file (binary status", binaryStatus[0], "!= CL_SUCCESS)");
        return optional<cl::Kernel>{};
    }

    if(errorCode != CL_SUCCESS){
        Log<LogLevel::Error>("Failed to create OpenCL program from binary file (error code", binaryStatus[0], "!= CL_SUCCESS)");
        return optional<cl::Kernel>{};
    }

    // This call will get the kernel object from program. A kernel is an
    // OpenCL function that is executed on the FPGA.
    cl::Kernel kernel(program, kernelName.c_str(), &errorCode);

    if(errorCode != CL_SUCCESS){
        Log<LogLevel::Error>("Failed to create kernel with name", kernelName, "from program", xclbin);
        return optional<cl::Kernel>{};
    }

    return tl::make_optional(kernel);
}
