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
#include <iterator>

#include <CL/cl2.hpp>

#include <host/logger.hpp>

#include <optional.hpp>

using tl::optional;
using Logger::Log;
using Logger::LogLevel;

enum class Access { Read, Write, ReadWrite };

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

template<typename IteratorType, typename T>
constexpr bool IsIteratorOfType() {
  return std::is_same<typename std::iterator_traits<IteratorType>::value_type,
                      T>::value;
}

template<typename IteratorType>
constexpr bool IsRandomAccess() {
  return std::is_base_of<
      std::random_access_iterator_tag,
      typename std::iterator_traits<IteratorType>::iterator_category>::value;
}

template<typename T, Access access>
optional<cl::Buffer> MakeBuffer(const cl::Context &context, size_t numElements){
	cl_int errorCode; cl_mem_flags flags;

    switch (access) {
	  case Access::Read:
		flags = CL_MEM_READ_ONLY;
		break;
	  case Access::Write:
		flags = CL_MEM_WRITE_ONLY;
		break;
	  case Access::ReadWrite:
		flags = CL_MEM_READ_WRITE;
		break;
	}

    cl::Buffer buffer(context, flags, numElements * sizeof(T), nullptr, &errorCode);

    if(errorCode != CL_SUCCESS){
    	Log<LogLevel::Error>("Failed to initialize device memory.");
    	return optional<cl::Buffer>{};
    }

    return tl::make_optional(buffer);
}

template<typename T>
bool SetKernelArguments(cl::Kernel &kernel, size_t index, T &&arg){
    cl_int errorCode = kernel.setArg(index, arg);
    if(errorCode != CL_SUCCESS){
        Log<LogLevel::Error>("Failed to set kernel argument", index);
        return false;
    }
    return true;
}

template<typename T, typename... Ts>
bool SetKernelArguments(cl::Kernel &kernel, size_t index, T &&arg, Ts &&... args){
    if(!SetKernelArguments(kernel, index, std::forward<T>(arg)))
        return false;
    return SetKernelArguments(kernel, index+1, std::forward<Ts>(args)...);
}

template<typename T, typename IteratorType, typename = typename
         std::enable_if<IsIteratorOfType<IteratorType, T>() && IsRandomAccess<IteratorType>()>::type>
bool CopyFromHost(cl::CommandQueue &queue, const cl::Buffer &buffer, IteratorType begin, IteratorType end){
    auto numElements = std::distance(begin, end);
    T *hostPtr = const_cast<T*>(&(*begin));

    // enqueueWriteBuffer() API call is a request to enqueue a write operation. This
    // API call does not immediately initiate the data transfer. The data transfer happens
    // when a kernel is enqueued which has the respective buffer as one of its arguments.
    cl_int errorCode = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, numElements * sizeof(T), hostPtr);

    if(errorCode != CL_SUCCESS){
        Log<LogLevel::Error>("Failed to copy data to device.");
        return false;
    }

    return true;
}

template<typename T, typename IteratorType, typename = typename
         std::enable_if<IsIteratorOfType<IteratorType, T>() && IsRandomAccess<IteratorType>()>::type>
bool CopyToHost(cl::CommandQueue &queue, const cl::Buffer &buffer, size_t numElements, IteratorType target){
    // Data can be transferred back to the host using the read buffer operation
    cl_int errorCode = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, numElements * sizeof(T), &(*target));

    if(errorCode != CL_SUCCESS){
        Log<LogLevel::Error>("Failed top copy data from device.");
        return false;
    }

    return true;
}
