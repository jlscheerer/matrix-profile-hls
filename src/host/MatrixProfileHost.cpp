#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <array>

#include "cxxopts.hpp"
#include "optional.hpp"
#include "host/Logger.hpp"

#include "MatrixProfile.hpp"
#include "host/OpenCL.hpp"
#include "host/MatrixProfileHost.hpp"

using tl::optional;
using Logger::Log;
using Logger::LogLevel;

static const std::string versionName{"0.0.1 - Host Skeleton"};

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

int RunMatrixProfileKernel(std::string xclbin, std::string input, optional<std::string> output){
    cl::Device device;
    if(optional<cl::Device> opt = FindDevice())
    	device = *opt;
    else return EXIT_FAILURE;

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    cl::Kernel kernel;
    if(optional<cl::Kernel> opt = MakeKernel(context, device, xclbin, "MatrixProfileKernelTLF"))
    	kernel = *opt;
    else return EXIT_FAILURE;

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    cl::Buffer buffer_T, buffer_MP, buffer_MPI;
    if(optional<cl::Buffer> opt = MakeBuffer<int, Access::Read>(context, n))
        buffer_T = *opt;
    else return EXIT_FAILURE;

    if(optional<cl::Buffer> opt = MakeBuffer<int, Access::Write>(context, rs_len))
        buffer_MP = *opt;
    else return EXIT_FAILURE;

    if(optional<cl::Buffer> opt = MakeBuffer<int, Access::Write>(context, rs_len))
        buffer_MPI = *opt;
    else return EXIT_FAILURE;

    // TODO load actual input file containg time series
    
    std::array<data_t, n> host_T;
    host_T[0] = 1; host_T[1] = 4; host_T[2] = 9; host_T[3] = 16; host_T[4] = 25; host_T[5] = 36; host_T[6] = 49; host_T[7] = 64;

    std::array<data_t, rs_len> host_MP;
    std::array<index_t, rs_len> host_MPI;

    // Copy Time Series to the FPGA
    CopyFromHost<data_t>(queue, buffer_T, host_T.cbegin(), host_T.cend());

    // Set the Kernel Arguments
    SetKernelArguments(kernel, 0, n, m, buffer_T, buffer_MP, buffer_MPI);

    // Launch the Kernel & wait for it to finish
    queue.enqueueTask(kernel);
    queue.finish();

    // Read resulting Matrix Profile and Matrix Profile Index
    CopyToHost<data_t>(queue, buffer_MP, rs_len, host_MP.data());
    CopyToHost<index_t>(queue, buffer_MPI, rs_len, host_MPI.data());

    queue.finish();

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

        Log<LogLevel::Info>(xclbin, input);

        return RunMatrixProfileKernel(xclbin, input, output);
    }catch(const cxxopts::option_not_exists_exception&){
        Log<LogLevel::Error>("Unknown argument\n");
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }
}
