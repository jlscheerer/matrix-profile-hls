#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <array>

#include "cxxopts.hpp"
#include "optional.hpp"
#include "host/Logger.hpp"

#include "host/OpenCL.hpp"
#include "host/MatrixProfileHost.hpp"

#include "MatrixProfile.hpp"

using tl::optional;
using Logger::Log;
using Logger::LogLevel;

const std::string versionName{"0.0.1 - Host Skeleton"};

static const int DATA_SIZE = 4096;

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
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    cl::Kernel kernel;
    if(optional<cl::Kernel> opt = MakeKernel(context, device, xclbin, "MatrixProfileKernelTLF"))
    	kernel = *opt;
    else return EXIT_FAILURE;

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    cl::Buffer buffer_a, buffer_b, buffer_result;
    if(optional<cl::Buffer> opt = MakeBuffer<int, Access::Read>(context, DATA_SIZE))
        buffer_a = *opt;
    else return EXIT_FAILURE;

    if(optional<cl::Buffer> opt = MakeBuffer<int, Access::Read>(context, DATA_SIZE))
        buffer_b = *opt;
    else return EXIT_FAILURE;

    if(optional<cl::Buffer> opt = MakeBuffer<int, Access::Write>(context, DATA_SIZE))
        buffer_result = *opt;
    else return EXIT_FAILURE;

    std::array<int, DATA_SIZE> A, B, C;
    for(int i = 0; i < DATA_SIZE; ++i){
        A[i] = 10;
        B[i] = 20;
        C[i] = 0;
    }

    CopyFromHost<int>(q, buffer_a, A.cbegin(), A.cend());
    CopyFromHost<int>(q, buffer_b, B.cbegin(), B.cend());

    // Set the kernel Arguments
    SetKernelArguments(kernel, 0, buffer_a, buffer_b, buffer_result, DATA_SIZE);

    //Launch the Kernel & wait for it to finish
    q.enqueueTask(kernel);
    q.finish();

    // Data can be transferred back to the host using the read buffer operation
    CopyToHost<int>(q, buffer_result, DATA_SIZE, C.data());

    //Verify the result
    int match = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        int host_result = A[i] + B[i];
        if (C[i] != host_result) {
            printf(error_message.c_str(), i, host_result, C[i]);
            match = 1;
            break;
        }
    }

    q.finish();

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
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
