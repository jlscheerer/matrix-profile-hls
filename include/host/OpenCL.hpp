/**
 * @file    OpenCL.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   OpenCL Wrapper (Context, Buffer<T, Access>, Program, Kernel)
 */

#pragma once

// Xilinx provided Configuration Values
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <string>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <chrono>
#include <stdexcept>

#include <CL/cl2.hpp>

#include "host/Timer.hpp"
#include "host/Logger.hpp"

#include "optional.hpp"

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

namespace OpenCL{

    class ConfigurationError: public std::logic_error{
    public:
        ConfigurationError(std::string const &message): std::logic_error(message) {}

        ConfigurationError(char const *const message): std::logic_error(message) {}
    };

    class RuntimeError: public std::runtime_error {
    public:
        RuntimeError(std::string const &message): std::runtime_error(message) {}

        RuntimeError(char const *const message): std::runtime_error(message) {}
    };

    enum class Access { ReadOnly, WriteOnly, ReadWrite };

    class Program;

    class Kernel;

    template<typename T, Access access>
    class Buffer;

    class Context{
        public:
            Context();

            Program MakeProgram(const std::string &xclbin);

            template<typename T, Access access>
            Buffer<T, access> MakeBuffer(size_t size);

            // Returns the internal OpenCL command queue.
            inline cl::CommandQueue const &commandQueue() const { return m_queue; }

            // Returns the internal OpenCL device
            inline cl::Device const &device() const { return m_device; }

            // Returns the internal OpenCL context
            inline cl::Context const &context() const { return m_context; }

            ~Context();
        private:
            cl::Context m_context;
            cl::CommandQueue m_queue;
            cl::Device m_device;
            static cl::Device FindDevice();
    };

    template<typename T, Access access>
    class Buffer{
        public:
            Buffer(Context &context, size_t size): m_context(&context), m_size(size){
                cl_int errorCode; cl_mem_flags flags;

                switch (access) {
                    case Access::ReadOnly:
                        flags = CL_MEM_READ_ONLY;
                        break;
                    case Access::WriteOnly:
                        flags = CL_MEM_WRITE_ONLY;
                        break;
                    case Access::ReadWrite:
                        flags = CL_MEM_READ_WRITE;
                        break;
                }

                m_buffer = cl::Buffer(context.context(), flags, m_size * sizeof(T), nullptr, &errorCode);

                if(errorCode != CL_SUCCESS)
                    throw RuntimeError("Failed to initialize device memory.");
            }

            template<typename IteratorType, typename = typename
                     std::enable_if<IsIteratorOfType<IteratorType, T>() && IsRandomAccess<IteratorType>()>::type>
            void CopyFromHost(IteratorType begin, IteratorType end){
                T *hostPtr = const_cast<T*>(&(*begin));

                // enqueueWriteBuffer() API call is a request to enqueue a write operation. This
                // API call does not immediately initiate the data transfer. The data transfer happens
                // when a kernel is enqueued which has the respective buffer as one of its arguments.
                cl_int errorCode = m_context->commandQueue().enqueueWriteBuffer(m_buffer, CL_TRUE, 0, m_size * sizeof(T), hostPtr);

                if(errorCode != CL_SUCCESS)
                    throw RuntimeError("Failed to copy data to device.");
            }

            template<typename IteratorType, typename = typename
                     std::enable_if<IsIteratorOfType<IteratorType, T>() && IsRandomAccess<IteratorType>()>::type>
            void CopyToHost(IteratorType target){
                // Data can be transferred back to the host using the read buffer operation
                cl_int errorCode = m_context->commandQueue().enqueueReadBuffer(m_buffer, CL_TRUE, 0, m_size * sizeof(T), &(*target));

                if(errorCode != CL_SUCCESS)
                    throw RuntimeError("Failed top copy data from device.");
            }

            // Returns the internal OpenCL context
            inline cl::Buffer const &buffer() const { return m_buffer; }

            ~Buffer(){}

        private:
            Context *m_context;
            size_t m_size;
            cl::Buffer m_buffer;
    };

    class Program{
        public:
            Program(Context &context, const std::string &xclbin);

            template<typename... Ts>
            Kernel MakeKernel(const std::string &name, Ts &&... args);

            // Returns the internal OpenCL program
            inline cl::Program const &program() const { return m_program; }

            // Returns the internal OpenCL command queue of the current context
            inline cl::CommandQueue const &commandQueue() const { return m_context->commandQueue(); }

            ~Program();
        private:
            Context *m_context;
            cl::Program m_program;
    };

    class Kernel{
        public:
            Kernel(Program &program, const std::string &name);

            std::chrono::nanoseconds ExecuteTask();

            ~Kernel();
        private:
            friend class Program;

            Program *m_program;
            cl::Kernel m_kernel;

            template<typename T, Access access>
            void SetKernelArguments(size_t index, Buffer<T, access> &arg){
                cl_int errorCode = m_kernel.setArg(index, arg.buffer());
                if(errorCode != CL_SUCCESS)
                    throw ConfigurationError("Failed to set kernel argument " + std::to_string(index));
            }

            template<typename T>
            void SetKernelArguments(size_t index, T &&arg){
                cl_int errorCode = m_kernel.setArg(index, arg);
                if(errorCode != CL_SUCCESS)
                    throw ConfigurationError("Failed to set kernel argument " + std::to_string(index));
            }

            void SetKernelArguments(size_t){}
            void SetKernelArguments(){}

            template<typename T, typename... Ts>
            void SetKernelArguments(size_t index, T &&arg, Ts &&... args){
                SetKernelArguments(index, std::forward<T>(arg));
                SetKernelArguments(index + 1, std::forward<Ts>(args)...);
            }
    };

    cl::Device Context::FindDevice(){
        // Traverse all available Platforms to find Xilinx Platform and targeted Device
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        cl_int errorCode;

        errorCode = cl::Platform::get(&platforms);
        if(errorCode != CL_SUCCESS)
            throw ConfigurationError("Failed to retrieve OpenCL platforms");

        for(auto &platform: platforms){
            std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&errorCode);
            if(errorCode != CL_SUCCESS)
                throw ConfigurationError("Failed to retrieve platform vendor name");

            if(platformName == "Xilinx"){
                platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
                if(devices.size())
                    return devices[0];
            }
        }
        throw ConfigurationError("Failed to find Xilinx-Platform or no available devices found");
    }

    Context::Context(){
        m_device = FindDevice();
        m_context = cl::Context(m_device);
        // TODO: disable profiling for benchmarks
        m_queue = cl::CommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE);
    }

    template<typename T, Access access>
    Buffer<T, access> Context::MakeBuffer(size_t size){
        return Buffer<T, access>{*this, size};
    }

    Program Context::MakeProgram(const std::string &xclbin){
        return Program{*this, xclbin};
    }

    Context::~Context(){
        // finish any ongoing tasks
        m_queue.finish();
    }

    Program::Program(Context &context, const std::string &xclbin): m_context(&context){
        std::ifstream bin(xclbin, std::ios::in | std::ios::binary | std::ios::ate);

        if(!bin.is_open())
            throw RuntimeError("Failed to open program binary file (.xclbin)");

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
        devices.emplace_back(context.device());

        // Create Program from Binary File
        m_program = cl::Program(context.context(), devices, binaries, &binaryStatus, &errorCode);

        if(binaryStatus[0] != CL_SUCCESS)
            throw RuntimeError("Failed to create OpenCL program from binary file (binary status " + std::to_string(binaryStatus[0]) + " != CL_SUCCESS)");

        if(errorCode != CL_SUCCESS)
            throw RuntimeError("Failed to create OpenCL program from binary file (error code " + std::to_string(errorCode) + " != CL_SUCCESS)");

        bin.close();
    }

    template<typename... Ts>
    Kernel Program::MakeKernel(const std::string &name, Ts &&... args){
        Kernel kernel{*this, name};
        kernel.SetKernelArguments(0, std::forward<Ts>(args)...);
        return kernel;
    }

    Program::~Program(){}

    Kernel::Kernel(Program &program, const std::string &name): m_program(&program) {
        cl_int errorCode;
        // This call will get the kernel object from the program binary.
        // A kernel is an OpenCL function that is executed on the FPGA.
        m_kernel = cl::Kernel(program.program(), name.c_str(), &errorCode);

        if(errorCode != CL_SUCCESS)
            throw RuntimeError("Failed to create kernel with name '" + name + "'");
    }

    std::chrono::nanoseconds Kernel::ExecuteTask(){
        m_program->commandQueue().enqueueTask(m_kernel);
        // Record time required for execution
        Timer timer;
        m_program->commandQueue().finish();
        // Return the execution time
        return timer.Elapsed();
    }

    Kernel::~Kernel(){}

};
