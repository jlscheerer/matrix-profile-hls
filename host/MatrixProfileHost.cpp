/**
 * @file    MatrixProfileHost.cpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Host-Application (C++/OpenCL)
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <array>
#include <chrono>

#include "cxxopts.hpp"
#include "optional.hpp"

#include "Config.hpp"

#include "host/MatrixProfileHost.hpp"
#include "host/OpenCL.hpp"
#include "host/FileIO.hpp"
#include "host/Timer.hpp"
#include "host/Logger.hpp"

using tl::optional;
using Logger::Log;
using Logger::LogLevel;

using OpenCL::Access;
using OpenCL::MemoryBank;

/**
 * @param xclbin full path to the (.xclbin) binary
 * @param input  input (time series) file name (without extension), located under data/binary/
 * @param output output (matrix profile/matrix profile index) file name (without extension); if specified
 *               the result will be stored as a .mpb (matrix profile) and a .mpib (matrix profile index) file
 * @return int   EXIT_SUCCESS in the case of a sucessful execution and EXIT_FAILURE otherwise
 */
int RunMatrixProfileKernel(const std::string &xclbin, const std::string &input, const optional<std::string> &output){
    // Allocate Host-Side Memory
    std::array<data_t, n> host_T;
    std::array<data_t, sublen> host_MP;
    std::array<index_t, sublen> host_MPI;

    // cwd: /media/sd-mmcblk0p1
    // Load Input File Containing Time Series Data into Host Memory
    Log<LogLevel::Verbose>("Loading input time series...");
    if(!FileIO::ReadBinaryFile(input, host_T))
        return EXIT_FAILURE;

    Log<LogLevel::Verbose>("Initializing OpenCL context...");
    OpenCL::Context context;

    // These commands will allocate memory on the Device. OpenCL::Buffer 
    // objects can be used to reference the memory locations on the device.
    Log<LogLevel::Verbose>("Initializing Memory...");
    OpenCL::Buffer<data_t, Access::ReadOnly> buffer_T{
        context.MakeBuffer<data_t, Access::ReadOnly>(MemoryBank::MemoryBank0, n)
    };
    OpenCL::Buffer<data_t, Access::WriteOnly> buffer_MP{
        context.MakeBuffer<data_t, Access::WriteOnly>(MemoryBank::MemoryBank0, sublen)
    };
    OpenCL::Buffer<index_t, Access::WriteOnly> buffer_MPI{
        context.MakeBuffer<index_t, Access::WriteOnly>(MemoryBank::MemoryBank1, sublen)
    };

    Log<LogLevel::Verbose>("Programming device...");
    OpenCL::Program program{context.MakeProgram(xclbin)};

    Log<LogLevel::Verbose>("Copying memory to device...");
    buffer_T.CopyFromHost(host_T.cbegin(), host_T.cend());

    Log<LogLevel::Verbose>("Creating Kernel...");
    OpenCL::Kernel kernel{
        program.MakeKernel(KernelTLF, buffer_T, buffer_MP, buffer_MPI)
    };

    Log<LogLevel::Verbose>("Executing Kernel...");
    std::chrono::nanoseconds executionTime{
        kernel.ExecuteTask()
    };

    Log<LogLevel::Info>("Kernel completed successfully in", executionTime);

    Log<LogLevel::Verbose>("Copying back result...");
    buffer_MP.CopyToHost(host_MP.data());
    buffer_MPI.CopyToHost(host_MPI.data());

    if(output){
        Log<LogLevel::Verbose>("Saving results (MP/MPI) to file...");
        // Write the Matrix Profile to disk
        if(!FileIO::WriteBinaryFile((*output) + ".mpb", host_MP))
            return EXIT_FAILURE;

        // Write the Matrix Profile Index to disk
        if(!FileIO::WriteBinaryFile((*output) + ".mpib", host_MPI))
            return EXIT_FAILURE;
    }else{
        // Just output the result to the console (for debugging)
        std::cout << "MP:";
        for(size_t i = 0; i < sublen; ++i)
    	    std::cout << " " << host_MP[i];
        std::cout << std::endl;

        std::cout << "MPI:";
        for(size_t i = 0; i < sublen; ++i)
            std::cout << " " << host_MPI[i];
        std::cout << std::endl;
    }

    Log<LogLevel::Verbose>("Terminating Host.");

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    std::string hostName{argv[0]};
    cxxopts::Options options(hostName, "Matrix Profile Host - (C++/OpenCL)");

    options.add_options()
        ("b,xclbin", ".xclbin to load as the kernel [required]", cxxopts::value<std::string>())
        // specify the input time series to send to the kernel
        ("i,input", "input file (time series) [required]", cxxopts::value<std::string>())
        // specify the output file path
        ("o,output", "output file (matrix profile / matrix profile index)", cxxopts::value<std::string>())
        // enable verbose output, e.g. show "Initializing OpenCL context..."
        ("verbose", "increase output verbosity")
        ("v,version", "prints version information and exits")
        ("h,help", "shows help message and exits");

    try{
        auto args{options.parse(argc, argv)};

        if(args.count("help")){
            std::cout << options.help() << std::endl;
            return EXIT_SUCCESS;
        }

        if(args.count("version")){
            std::cout << versionName << std::endl;
            return EXIT_SUCCESS;
        }

        if(!args.count("xclbin")){
            Log<LogLevel::Error>("--xclbin required\n");
            std::cout << options.help() << std::endl;
            return EXIT_FAILURE;
        }

        if(!args.count("input")){
            Log<LogLevel::Error>("--input required\n");
            std::cout << options.help() << std::endl;
            return EXIT_FAILURE;
        }

        Logger::Verbose = args.count("verbose");

        std::string xclbin{args["xclbin"].as<std::string>()};
        std::string input{args["input"].as<std::string>()};

        optional<std::string> output{args.count("output") 
                                     ? tl::make_optional(args["output"].as<std::string>()) 
                                     : optional<std::string>{} };

        return RunMatrixProfileKernel(xclbin, input, output);
    }catch(const cxxopts::option_not_exists_exception&){
        Log<LogLevel::Error>("Unknown argument\n");
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }
}
