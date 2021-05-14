#include <cstdlib>
#include <fstream>
#include <iostream>
#include <array>
#include <chrono>

#include "cxxopts.hpp"
#include "optional.hpp"

#include "MatrixProfile.hpp"

#include "host/MatrixProfileHost.hpp"
#include "host/OpenCL.hpp"
#include "host/FileIO.hpp"
#include "host/Timer.hpp"
#include "host/Logger.hpp"

using tl::optional;
using Logger::Log;
using Logger::LogLevel;

using OpenCL::Access;

int RunMatrixProfileKernel(std::string xclbin, std::string input, optional<std::string> output){
    // Allocate Host-Side Memory
    std::array<data_t, n> host_T;
    std::array<data_t, rs_len> host_MP;
    std::array<index_t, rs_len> host_MPI;

    // TODO: Actually respect the provided input parameter
    // cwd: /media/sd-mmcblk0p1
    // Load Input File Containing Time Series Data into Host Memory
    if(!FileIO::ReadBinaryFile<data_t>("data/binary/small8_syn.tsb", host_T))
        return EXIT_FAILURE;

    OpenCL::Context context;

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    OpenCL::Buffer<data_t, Access::ReadOnly> buffer_T{
        context.MakeBuffer<data_t, Access::ReadOnly>(n)
    };
    OpenCL::Buffer<data_t, Access::WriteOnly> buffer_MP{
        context.MakeBuffer<data_t, Access::WriteOnly>(rs_len)
    };
    OpenCL::Buffer<index_t, Access::WriteOnly> buffer_MPI{
        context.MakeBuffer<index_t, Access::WriteOnly>(rs_len)
    };

    OpenCL::Program program{context.MakeProgram(xclbin)};

    OpenCL::Kernel kernel{
        program.MakeKernel(KernelTLF, buffer_T, buffer_MP, buffer_MPI)
    };

    buffer_T.CopyFromHost(host_T.cbegin(), host_T.cend());

    std::chrono::nanoseconds executionTime{kernel.ExecuteTask()};

    Log<LogLevel::Info>("Kernel completed successfully in", executionTime);

    buffer_MP.CopyToHost(host_MP.data());
    buffer_MPI.CopyToHost(host_MPI.data());

    // TODO: Actually respect the provided output parameter
    // Write the Matrix Profile to disk
    if(!FileIO::WriteBinaryFile<data_t>("output.mpb", host_MP))
        return EXIT_FAILURE;

    // Write the Matrix Profile Index to disk
    if(!FileIO::WriteBinaryFile<index_t>("output.mpib", host_MPI))
        return EXIT_FAILURE;

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
