cmake_minimum_required(VERSION 3.0)
project(MatrixProfileHLS)

set(CMAKE_CXX_STANDARD 11)

# Matrix Profile Computation Options
set(MP_DATA_TYPE "double" CACHE STRING "type for time series & resulting matrix profile")
set(MP_INDEX_TYPE "int" CACHE STRING "type for the resulting matrix profile index")

set(MP_SIZE_N 8 CACHE STRING "length of the time series data")
set(MP_SIZE_M 4 CACHE STRING "subsequence length for the matrix profile")
set(MP_SIZE_T 4 CACHE STRING "tile-size (only applicable for Stream2D) [assumption: t ≥ m]")

set(MP_KERNEL "Stream2D" CACHE STRING "implementation to use for the matrix profile kernel {Streamless, Stream1D, Stream2D}")

set(MP_AGGREGATE_INIT "-1e12" CACHE STRING "\"negative infinity\" used to initialize aggregates")
set(MP_INDEX_INIT "-1" CACHE STRING "used to indicate an invalid/undetermined index")

# Custom option to enable software testing using google-test
# Not enabled by default has to be explicitly enabled via -DBUILD_TESTS=ON
option(BUILD_TESTS "Build software tests (google-test)" OFF)

# Create the configuration file
configure_file(include/Config.hpp.in Config.hpp)

# TODO: Check validity of parameters!

set(HOST_COMPILER "aarch64-linux-gnu-g++")
set(VITIS_COMPILER "/opt/xilinx/vitis/Vitis/2020.1/bin/v++")

# Host Application (Compile & Link)
set(HOST_COMPILE_COMMAND
    -std=c++0x
    -DVITIS_PLATFORM=xilinx_zcu102_base_202010_1
    -D__USE_XOPEN2K8
    -I"${CMAKE_CURRENT_BINARY_DIR}"
    -I/opt/xilinx/vitis/Vivado/2020.1/include/
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
    -I/opt/xilinx/petalinux/sysroots/aarch64-xilinx-linux/usr/include/xrt/
    -O0 -g -Wall -c -fmessage-length=0
    --sysroot=/opt/xilinx/petalinux/sysroots/aarch64-xilinx-linux)

set(HOST_LINK_COMMAND
    -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++
    -L/opt/xilinx/petalinux/sysroots/aarch64-xilinx-linux/usr/lib/
    --sysroot=/opt/xilinx/petalinux/sysroots/aarch64-xilinx-linux)

add_custom_target(host COMMAND ${HOST_COMPILER} ${HOST_COMPILE_COMMAND} -o "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o" "${CMAKE_CURRENT_SOURCE_DIR}/host/MatrixProfileHost.cpp"
                       COMMAND ${HOST_COMPILER} -o "${CMAKE_CURRENT_BINARY_DIR}/matrix-profile-hls" "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o" ${HOST_LINK_COMMAND})

# Kernel (Compile & Link)
set(KERNEL_COMPILE_COMMAND
    --compile 
    -I"${CMAKE_CURRENT_BINARY_DIR}"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/kernel"
    --config "${CMAKE_CURRENT_SOURCE_DIR}/config/common-config.cfg"
    --config "${CMAKE_CURRENT_SOURCE_DIR}/config/compile-config.cfg"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include")

set(KERNEL_LINK_COMMAND
    --link
    --config "${CMAKE_CURRENT_SOURCE_DIR}/config/common-config.cfg"
    --config "${CMAKE_CURRENT_SOURCE_DIR}/config/link-config.cfg")

add_custom_target(compile COMMAND ${VITIS_COMPILER} --target hw_emu ${KERNEL_COMPILE_COMMAND} -o"${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo" "${CMAKE_CURRENT_SOURCE_DIR}/kernel/MatrixProfileKernel${MP_KERNEL}.cpp")

add_custom_target(link COMMAND ${VITIS_COMPILER} --target hw_emu ${KERNEL_LINK_COMMAND} -o"${CMAKE_CURRENT_BINARY_DIR}/mp_binary_container.xclbin" "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo")

# Package SD-Card
set(PACKAGE_COMMAND
    --package
    --config "${CMAKE_CURRENT_SOURCE_DIR}/config/package-config.cfg"
    --package.sd_dir "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_custom_target(package_sd COMMAND ${VITIS_COMPILER} --target hw_emu ${PACKAGE_COMMAND} "${CMAKE_CURRENT_BINARY_DIR}/mp_binary_container.xclbin")

# Software Testing
if (BUILD_TESTS)

    # make sure to clone respositore with --recursive
    # or download via git submodule update --init 
    add_subdirectory(googletest)

    enable_testing()

    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    include_directories(${PROJECT_SOURCE_DIR}/kernel ${PROJECT_SOURCE_DIR}/test/include)

    add_executable(runStreamlessKernelTests "${PROJECT_SOURCE_DIR}/test/TestStreamlessKernel.cpp")
    target_link_libraries(runStreamlessKernelTests gtest gtest_main)

    add_executable(runStream1DKernelTests "${PROJECT_SOURCE_DIR}/test/TestStream1DKernel.cpp")
    target_link_libraries(runStream1DKernelTests gtest gtest_main)

    add_executable(runStream2DKernelTests "${PROJECT_SOURCE_DIR}/test/TestStream2DKernel.cpp")
    target_link_libraries(runStream2DKernelTests gtest gtest_main)

    # This is so you can do 'make test' to see all tests run, instead of
    # manually running the executable runUnitTests to see those specific tests.
    add_test(NAME TestStreamlessKernel COMMAND runStreamlessKernelTests)
    add_test(NAME TestStream1DKernel COMMAND runStream1DKernelTests)
    add_test(NAME TestStream2DKernel COMMAND runStream2DKernelTests)

endif()