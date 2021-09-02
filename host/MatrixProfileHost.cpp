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
#include "host/HostSideComputation.hpp"

#include "cmath"

#include "host/OpenCL.hpp"
#include "host/FileIO.hpp"
#include "host/BenchmarkProfile.hpp"
#include "host/Logger.hpp"

using tl::optional;
using Logger::Log;
using Logger::LogLevel;

using OpenCL::Access;
using OpenCL::MemoryBank;

// Allocate Host-Side Memory (needs to be statically allocated!)
static std::array<double, n> host_T;
static std::array<InputDataPack, n - m + 1> host_input;
static std::array<std::array<OutputDataPack, n - m + 1>, kNumKernels> host_output;

// Intermediate Results Storing column- & row-wise aggregates
static std::array<aggregate_t, n - m + 1> rowAggregates, columnAggregates;

// Resulting Matrix Profile and corresponding Matrix Profile Index
static std::array<double, n - m + 1> MP;
static std::array<index_t, n - m + 1> MPI;

/**
 * @param xclbin full path to the (.xclbin) binary
 * @param input  input (time series) file name (without extension), located under data/binary/
 * @param output output (matrix profile/matrix profile index) file name (without extension); if specified
 *               the result will be stored as a .mpb (matrix profile) and a .mpib (matrix profile index) file
 * @return int   EXIT_SUCCESS in the case of a sucessful execution and EXIT_FAILURE otherwise
 */
int RunMatrixProfileKernel(const std::string &xclbin, const std::string &input, const optional<std::string> &output){
    BenchmarkProfile profile;

    if (!output)
        Log<LogLevel::Warning>("No output (-o, --output) parameter provided. Results will be discarded!");

    // Load Input File Containing Time Series Data into Host Memory
    Log<LogLevel::Verbose>("Loading input time series...");
    if(!FileIO::ReadBinaryFile(input, host_T))
        return EXIT_FAILURE;

    Log<LogLevel::Info>("Pre-Computing Statistics on Host");
    HostSideComputation::PreComputeStatistics(profile, host_T, host_input);

    Log<LogLevel::Verbose>("Initializing OpenCL context...");
    OpenCL::Context context;

    Log<LogLevel::Verbose>("Initializing Memory...");

    // These commands will allocate memory on the Device. OpenCL::Buffer
    // objects can be used to reference the memory locations on the device.
    std::vector<OpenCL::Buffer<InputDataPack, Access::ReadOnly>> bufferInput;
    std::vector<OpenCL::Buffer<OutputDataPack, Access::WriteOnly>> bufferOutput;

    for (index_t i = 0; i < kNumKernels; ++i) {
        const MemoryBank bank = static_cast<MemoryBank>(i);
        bufferInput.push_back(context.MakeBuffer<InputDataPack, Access::ReadOnly>(bank, n - m + 1));
        bufferOutput.push_back(context.MakeBuffer<OutputDataPack, Access::WriteOnly>(bank, n - m + 1));
    }

    Log<LogLevel::Verbose>("Programming device...");
    OpenCL::Program program{context.MakeProgram(xclbin)};

    Log<LogLevel::Verbose>("Copying memory to device...");

    for (index_t i = 0; i < kNumKernels; ++i)
        bufferInput[i].CopyFromHost(host_input.cbegin(), host_input.cend());
    // After enqueing write operations, we can now actually perform them
    context.Finish();

    // Create kNumKernel Kernel Instances
    std::vector<OpenCL::Kernel> kernels;
    for (index_t i = 0; i < kNumKernels; ++i)
        kernels.emplace_back(program, KernelTLF, i + 1);

    Log<LogLevel::Verbose>("Starting Kernel Execution(s)...");

    constexpr index_t nIterations = (n - m + nColumns) / nColumns;
    for (index_t iteration = 0; iteration < nIterations; ++iteration) {
        const index_t nOffset = iteration * nColumns;
        const index_t nRows = n - m + 1 - nOffset;

        // Cyclically reference different Kernels
        OpenCL::Kernel &kernel = kernels[iteration % kNumKernels];

        // Specify Kernel Arguments for the current Iteration
        kernel.SetKernelArguments(0, n, m, iteration, bufferInput[iteration % kNumKernels],
                                  bufferOutput[iteration % kNumKernels]);

        kernel.EnqueueTask();

        // Copy back the intermediate result (enqueue)
        bufferOutput[iteration % kNumKernels].CopyToHost(host_output[iteration % kNumKernels].data(), nRows);

        // Once all Kernel jobs have been enqeued we finish the
        // Iteration and Process the Results
        if (iteration % kNumKernels == kNumKernels - 1) {
            profile.Push("2. FPGA Computation [" + (std::string(KERNEL_IMPL_NAME)) + ", w=" + std::to_string(w) + "]",
                         KernelTLF + " [iteration=" + std::to_string(iteration - (kNumKernels - 1)) + "-" + std::to_string(iteration) + "]", context.Finish());
            // Process the results (Could be done asychronously!) by integrating
            // them into the local aggregate buffers
            for (index_t i = kNumKernels - 1; i >= 0; --i)
                HostSideComputation::UpdateAggregates(profile, iteration - i, host_output, rowAggregates, columnAggregates);
        }
    }

    // In case kNumKernels does not divide number of iterations
    // perform work for excess iterations (as tasks have not been finished)
    if (nIterations % kNumKernels != 0) {
        profile.Push("2. FPGA Computation [" + (std::string(KERNEL_IMPL_NAME)) + ", w=" + std::to_string(w) + "]",
                     KernelTLF + " [iteration=" + std::to_string(nIterations - ((nIterations % kNumKernels) - 1)) + "-" + std::to_string(nIterations - 1) + "]", context.Finish());
        for (index_t i = nIterations % kNumKernels - 1; i >= 0; --i)
            HostSideComputation::UpdateAggregates(profile, nIterations - 1 - i, host_output, rowAggregates, columnAggregates);
    }

    Log<LogLevel::Info>("Kernel Execution Completed Successfully.");

    Log<LogLevel::Info>("Performing Post-Computation on Host");
    HostSideComputation::PostComputeAggregates(profile, rowAggregates, columnAggregates, MP, MPI);

    if (output) {
        Log<LogLevel::Verbose>("Saving results (MP/MPI) to file...");
        // Write the Matrix Profile to disk
        if(!FileIO::WriteBinaryFile((*output) + ".mpb", MP))
            return EXIT_FAILURE;

        // Write the Matrix Profile Index to disk
        if(!FileIO::WriteBinaryFile((*output) + ".mpib", MPI))
            return EXIT_FAILURE;
    }

    profile.Report();

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

    try {
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
                                     : optional<std::string>{}};

        return RunMatrixProfileKernel(xclbin, input, output);
    } catch(const cxxopts::option_not_exists_exception&) {
        Log<LogLevel::Error>("Unknown argument\n");
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }
}
