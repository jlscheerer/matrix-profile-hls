cmake_minimum_required(VERSION 3.0)

# set to NONE to avoid/skip compiler checks
project(MatrixProfileHLS NONE)

set(CMAKE_CXX_STANDARD 11)

# Matrix Profile Target Options
set(MP_PLATFORM "xilinx_zcu102_base_202010_1" CACHE STRING "Platform for Vitis (--platform)")
# https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/embeddedplatforms.html
option(MP_EMBEDDED "build for Vitis Embedded Platforms" ON)

# Options for Embedded Platforms (Embedded acceleration requires a valid sysroot location)
set(MP_SYSROOT "/opt/xilinx/petalinux/sysroots/aarch64-xilinx-linux" CACHE STRING "Application settings: Sysroot path")
set(MP_ROOT_FS "/u/home/raoofy/xilinx-zynqmp-common-v2020.1/rootfs.ext4" CACHE STRING "Application settings: ROOT FS")
set(MP_KERNEL_IMAGE "/u/home/raoofy/xilinx-zynqmp-common-v2020.1/Image" CACHE STRING "Application settings: Kernel Image")

# Matrix Profile Computation Options
set(MP_DATA_TYPE "double" CACHE STRING "type for time series & resulting matrix profile")
set(MP_INDEX_TYPE "int" CACHE STRING "type for the resulting matrix profile index")

set(MP_SIZE_N 8 CACHE STRING "length of the time series data")
set(MP_SIZE_M 4 CACHE STRING "subsequence length for the matrix profile")
set(MP_SIZE_T 4 CACHE STRING "tile-size (only applicable for Stream2D) [assumption: t ≥ m]")

set(MP_KERNEL "Stream2D" CACHE STRING "implementation to use for the matrix profile kernel {Streamless, Stream1D, Stream2D}")

set(MP_TARGET "hw_emu" CACHE STRING "compile target: {sw_emu, hw_emu, hw}")

# Custom option to enable software testing using google-test
# Not enabled by default has to be explicitly enabled via -DBUILD_TESTS=ON
option(BUILD_TESTS "Build software tests (google-test)" OFF)

# Create the configuration file
configure_file(include/Config.hpp.in Config.hpp)

# Configer v++ configuration files
configure_file(config/common-config.cfg.in config/common-config.cfg)
configure_file(config/compile-config.cfg.in config/compile-config.cfg)
configure_file(config/link-config.cfg.in config/link-config.cfg)

# Only Generate Platform Configuration for Embedded Platforms
if(MP_EMBEDDED)
    configure_file(config/package-config.cfg.in config/package-config.cfg)
endif()

# TODO: Check validity of parameters!

if(MP_EMBEDDED)
    set(HOST_COMPILER "aarch64-linux-gnu-g++")
else()
    set(HOST_COMPILER "g++")
endif()

set(VITIS_COMPILER "$ENV{XILINX_VITIS}/bin/v++")

# Host Application (Compile & Link)
if(MP_EMBEDDED)
    set(HOST_COMPILE_COMMAND
        -std=c++11
        -DVITIS_PLATFORM=${MP_PLATFORM}
        -D__USE_XOPEN2K8
        -I"${CMAKE_CURRENT_BINARY_DIR}"
        -I"$ENV{XILINX_VIVADO}/include/"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        -I${MP_SYSROOT}/usr/include/xrt/
        --sysroot=${MP_SYSROOT}
        -O3 -c)
else()
    set(HOST_COMPILE_COMMAND
        -std=c++11
        -DVITIS_PLATFORM=${MP_PLATFORM}
        -D__USE_XOPEN2K8
        -I"${CMAKE_CURRENT_BINARY_DIR}"
        -I"$ENV{XILINX_XRT}/include/"
        -I"$ENV{XILINX_VIVADO}/include/"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        -O3 -c)
endif()

if(MP_EMBEDDED)
    set(HOST_LINK_COMMAND
        -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++
        -L${MP_SYSROOT}/usr/lib/ --sysroot=${MP_SYSROOT})
else()
    set(HOST_LINK_COMMAND
        -L$ENV{XILINX_XRT}/lib/ -lOpenCL -lpthread -lrt -lstdc++)
endif()

add_custom_target(host COMMAND ${HOST_COMPILER} ${HOST_COMPILE_COMMAND} -o "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o" "${CMAKE_CURRENT_SOURCE_DIR}/host/MatrixProfileHost.cpp"
                       COMMAND ${HOST_COMPILER} -o "${CMAKE_CURRENT_BINARY_DIR}/matrix-profile-hls" "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o" ${HOST_LINK_COMMAND})

# Kernel (Compile & Link)
set(KERNEL_COMPILE_COMMAND
    --compile 
    -I"${CMAKE_CURRENT_BINARY_DIR}"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/kernel"
    --config "${CMAKE_CURRENT_BINARY_DIR}/config/common-config.cfg"
    --config "${CMAKE_CURRENT_BINARY_DIR}/config/compile-config.cfg"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include")

set(KERNEL_LINK_COMMAND
    --link
    --config "${CMAKE_CURRENT_BINARY_DIR}/config/common-config.cfg"
    --config "${CMAKE_CURRENT_BINARY_DIR}/config/link-config.cfg")

add_custom_target(compile COMMAND ${VITIS_COMPILER} --target ${MP_TARGET} ${KERNEL_COMPILE_COMMAND} -o"${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo" "${CMAKE_CURRENT_SOURCE_DIR}/kernel/MatrixProfileKernel${MP_KERNEL}.cpp")

add_custom_target(link COMMAND ${VITIS_COMPILER} --target ${MP_TARGET} ${KERNEL_LINK_COMMAND} -o"${CMAKE_CURRENT_BINARY_DIR}/mp_binary_container.xclbin" "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo")

# Packaging Only Relevant for Embedded Platforms
if(MP_EMBEDDED)
    # Package SD-Card
    set(PACKAGE_COMMAND
        --package
        --config "${CMAKE_CURRENT_BINARY_DIR}/config/package-config.cfg"
        --package.sd_dir "${CMAKE_CURRENT_SOURCE_DIR}/data")

    add_custom_target(package_sd COMMAND ${VITIS_COMPILER} --target ${MP_TARGET} ${PACKAGE_COMMAND} "${CMAKE_CURRENT_BINARY_DIR}/mp_binary_container.xclbin")
endif()

# Software Testing via GoogleTest (https://github.com/google/googletest)
if(BUILD_TESTS)
    # make sure to clone respositore with --recursive
    # or download via git submodule update --init 
    add_subdirectory(googletest)

    enable_testing()

    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    include_directories(${PROJECT_SOURCE_DIR}/kernel ${PROJECT_SOURCE_DIR}/test/include)

    add_executable(testStreamlessKernel "${PROJECT_SOURCE_DIR}/test/TestStreamlessKernel.cpp")
    target_link_libraries(testStreamlessKernel gtest gtest_main)

    add_executable(testStream1DKernel "${PROJECT_SOURCE_DIR}/test/TestStream1DKernel.cpp")
    target_link_libraries(testStream1DKernel gtest gtest_main)

    add_executable(testStream2DKernel "${PROJECT_SOURCE_DIR}/test/TestStream2DKernel.cpp")
    target_link_libraries(testStream2DKernel gtest gtest_main)

    # This is so you can do 'make test' to see all tests run, instead of
    # manually running the executable runUnitTests to see those specific tests.
    add_test(NAME TestStreamlessKernel COMMAND testStreamlessKernel)
    add_test(NAME TestStream1DKernel COMMAND testStream1DKernel)
    add_test(NAME TestStream2DKernel COMMAND testStream2DKernel)
endif()