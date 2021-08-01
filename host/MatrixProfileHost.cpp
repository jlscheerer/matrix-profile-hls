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

/**
 * @param xclbin full path to the (.xclbin) binary
 * @param input  input (time series) file name (without extension), located under data/binary/
 * @param output output (matrix profile/matrix profile index) file name (without extension); if specified
 *               the result will be stored as a .mpb (matrix profile) and a .mpib (matrix profile index) file
 * @return int   EXIT_SUCCESS in the case of a sucessful execution and EXIT_FAILURE otherwise
 */
int RunMatrixProfileKernel(const std::string &xclbin, const std::string &input, const optional<std::string> &output){    
    BenchmarkProfile profile;

    // Allocate Host-Side Memory
    std::array<double, n> host_T;
    std::array<InputDataPack, n - m + 1> host_input;
    std::array<OutputDataPack, n - m + 1> host_output;

    // Matrix Profile (Euclidean Distance)
    std::array<double, n - m + 1> host_MPE;

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

    // These commands will allocate memory on the Device. OpenCL::Buffer 
    // objects can be used to reference the memory locations on the device.
    Log<LogLevel::Verbose>("Initializing Memory...");
    OpenCL::Buffer<InputDataPack, Access::ReadOnly> buffer_columns {
        context.MakeBuffer<InputDataPack, Access::ReadOnly>(MemoryBank::MemoryBank0, n - m + 1)
    };
    OpenCL::Buffer<InputDataPack, Access::ReadOnly> buffer_rows {
        context.MakeBuffer<InputDataPack, Access::ReadOnly>(MemoryBank::MemoryBank1, n - m + 1)
    };
    OpenCL::Buffer<OutputDataPack, Access::WriteOnly> buffer_output {
        context.MakeBuffer<OutputDataPack, Access::WriteOnly>(MemoryBank::MemoryBank2, n - m + 1)
    };

    Log<LogLevel::Verbose>("Programming device...");
    OpenCL::Program program{context.MakeProgram(xclbin)};

    Log<LogLevel::Verbose>("Copying memory to device...");
    buffer_columns.CopyFromHost(host_input.cbegin(), host_input.cend());
    buffer_rows.CopyFromHost(host_input.cbegin(), host_input.cend());

    std::array<aggregate_t, n - m + 1> rowAggregates, columnAggregates;

    constexpr index_t nIterations = (n - m + nColumns) / nColumns;
    for (index_t iteration = 0; iteration < nIterations; ++iteration) {
        const index_t nOffset = iteration * nColumns;
        const index_t nRows = n - m + 1 - nOffset;

        OpenCL::Kernel kernel{
            program.MakeKernel(KernelTLF, n, m, iteration, buffer_columns, buffer_rows, buffer_output)
        };

        profile.Push("2. FPGA Computation [" + (std::string(KERNEL_IMPL_NAME)) + ", w=" + std::to_string(w) + "]", 
                     KernelTLF + " [iteration=" + std::to_string(iteration) + ", nRows=" + std::to_string(nRows) + "]", kernel.ExecuteTask());

        // Copy back the intermediate result
        buffer_output.CopyToHost(host_output.data(), nRows);

        // Update Local "copies" of Aggregates
        Timer timer;
        for (index_t i = 0; i < nRows; ++i) {
            aggregate_t prevRow = iteration > 0 ? rowAggregates[i] : aggregate_t_init;
            aggregate_t prevCol = iteration > 0 ? columnAggregates[i + nOffset] : aggregate_t_init;

            aggregate_t currRow = host_output[i].rowAggregate;
            aggregate_t currCol = host_output[i].columnAggregate;

            rowAggregates[i] = currRow.value > prevRow.value ? currRow : prevRow;
            columnAggregates[i + nOffset] = currCol.value > prevCol.value ? currCol : prevCol;
        }
        const auto time = timer.Elapsed();
        profile.Push("3. Host-Side [Aggregate-Merge]", "Aggregate_Merge_" + std::to_string(iteration), time);
    }

    std::array<double, n - m + 1> MP;
    std::array<index_t, n - m + 1> MPI;

    Log<LogLevel::Info>("Performing Post-Computation on Host");
    HostSideComputation::PostComputeAggregates(profile, rowAggregates, columnAggregates, MP, MPI);

    // Log<LogLevel::Info>("Converting Pearson Correlation to Euclidean Distance");
    // HostSideComputation::PearsonCorrelationToEuclideanDistance(host_MP, host_MPE);
    if(output){
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
                                     : optional<std::string>{} };

        return RunMatrixProfileKernel(xclbin, input, output);
    } catch(const cxxopts::option_not_exists_exception&) {
        Log<LogLevel::Error>("Unknown argument\n");
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }
}


