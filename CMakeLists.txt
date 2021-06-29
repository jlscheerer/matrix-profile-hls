cmake_minimum_required(VERSION 3.2)

project(MatrixProfileHLS)

CMAKE_POLICY(SET CMP0054 NEW)
CMAKE_POLICY(SET CMP0057 NEW)

set(CMAKE_CXX_STANDARD 11)

# Matrix Profile Target Options
set(MP_PLATFORM "xilinx_u250_xdma_201830_2" CACHE STRING "Platform for Vitis (--platform)")

# https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/embeddedplatforms.html
option(MP_EMBEDDED "build for Vitis Embedded Platforms" OFF)

set(VALID_TARGETS sw_emu hw_emu hw)
set(MP_TARGET "hw_emu" CACHE STRING "compile target: {sw_emu, hw_emu, hw}")
set_property(CACHE MP_TARGET PROPERTY STRINGS ${VALID_TARGETS})

set(MP_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu" CACHE STRING "LIBRARAY_PATH environment variable. Required for hw_emu, hw build targets (ignored if environment variable is set)")
option(MP_QUICK_BUILD "quick build option for v++" OFF)

# Options for Embedded Platforms (Embedded acceleration requires a valid sysroot location)
set(MP_SYSROOT "/opt/xilinx/petalinux/sysroots/aarch64-xilinx-linux" CACHE STRING "Application settings: Sysroot path")
set(MP_ROOT_FS "/u/home/raoofy/xilinx-zynqmp-common-v2020.1/rootfs.ext4" CACHE STRING "Application settings: ROOT FS")
set(MP_KERNEL_IMAGE "/u/home/raoofy/xilinx-zynqmp-common-v2020.1/Image" CACHE STRING "Application settings: Kernel Image")

# Matrix Profile Computation Options
set(VALID_KERNELS Streamless Stream1D Stream2D)
set(MP_KERNEL "Stream2D" CACHE STRING "implementation to use for the matrix profile kernel {Streamless, Stream1D, Stream2D}")
set_property(CACHE MP_KERNEL PROPERTY STRINGS ${VALID_KERNELS})

set(VALID_DATA_TYPES double float ap16_t ap24_t ap32_t ap64_t)
set(MP_DATA_TYPE "double" CACHE STRING "type for time series & resulting matrix profile")
set_property(CACHE MP_DATA_TYPE PROPERTY STRINGS ${VALID_DATA_TYPES})

set(VALID_INDEX_TYPES int)
set(MP_INDEX_TYPE "int" CACHE STRING "type for the resulting matrix profile index")
set_property(CACHE MP_INDEX_TYPE PROPERTY STRINGS ${VALID_INDEX_TYPES})

set(MP_SIZE_N 8 CACHE STRING "length of the time series data")
set(MP_SIZE_M 4 CACHE STRING "subsequence length for the matrix profile")
set(MP_SIZE_T 4 CACHE STRING "tile-size (only applicable for Stream2D) [assumption: t ≥ m]")

# Custom option to enable software testing using google-test
# Not enabled by default has to be explicitly enabled via -DBUILD_TESTS=ON
option(BUILD_TESTS "Build software tests (google-test)" OFF)

option(SKIP_CHECKS "Skip Environment Variable Checks" OFF)

# check parameter validity
list(FIND VALID_TARGETS ${MP_TARGET} target_index)
if(target_index EQUAL -1)
    message(FATAL_ERROR "The specified value for 'MP_TARGET' [\"${MP_TARGET}\"] is invalid.\nTarget has to be one of 'sw_emu', 'hw_emu' or 'hw'.")
endif()

list(FIND VALID_KERNELS ${MP_KERNEL} kernel_index)
if(kernel_index EQUAL -1)
    message(FATAL_ERROR "The specified value for 'MP_KERNEL' [\"${MP_KERNEL}\"] is invalid.\nKernel has to be one of 'Streamless', 'Stream1D' or 'Stream2D'.")
endif()

list(FIND VALID_DATA_TYPES ${MP_DATA_TYPE} data_type_index)
if(data_type_index EQUAL -1)
    message(FATAL_ERROR "The specified value for 'MP_DATA_TYPE' [\"${MP_DATA_TYPE}\"] is invalid.\ndata_t has to be one of 'double', 'float', 'ap16_t', 'ap24_t', 'ap32_t' or 'ap64_t'.")
endif()

list(FIND VALID_INDEX_TYPES ${MP_INDEX_TYPE} index_type_index)
if(index_type_index EQUAL -1)
    message(FATAL_ERROR "The specified value for 'MP_INDEX_TYPE' [\"${MP_INDEX_TYPE}\"] is invalid.\nindex_t has to be one of 'int'.")
endif()

if(NOT(${MP_SIZE_N} MATCHES "^[0-9]+$"))
    message(FATAL_ERROR "The specified value for 'MP_SIZE_N' [\"${MP_SIZE_N}\"] is invalid, n has to be numeric!")
endif()

if(NOT(${MP_SIZE_M} MATCHES "^[0-9]+$") OR ${MP_SIZE_N} LESS ${MP_SIZE_M})
    message(FATAL_ERROR "The specified value for 'MP_SIZE_M' [\"${MP_SIZE_M}\"] is invalid, m has to be numeric and ≤ n!")
endif()

if(("${MP_KERNEL}" STREQUAL "Stream2D") AND (NOT(${MP_SIZE_T} MATCHES "^[0-9]+$") OR ${MP_SIZE_T} LESS ${MP_SIZE_M}))
    message(FATAL_ERROR "The specified value for 'MP_SIZE_T' [\"${MP_SIZE_T}\"] is invalid, t has to be numeric and ≥ m!")
endif()
# /check parameter validity

if(DEFINED ENV{LIBRARY_PATH})
    # Override MP_LIBRARY_PATH if LIBRARAY_PATH environment variable is set explicitly
    set(MP_LIBRARY_PATH $ENV{LIBRARY_PATH})
endif()

# Check that all required Environment Variables are set
if(NOT(SKIP_CHECKS))
    if(NOT(DEFINED ENV{XILINX_VITIS} AND DEFINED ENV{XILINX_VIVADO}))
        message(FATAL_ERROR "Environment has not been setup correctly!\n"
                            "XILINX_VITIS and/or XILINX_VIVADO environment variable(s) are not set.\n"
                            "For further information see: https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/settingupvitisenvironment.html\n"
                            "Did you forget to run \"source <Vitis_install_path>/settings64.sh\"?")
    endif()

    if(NOT((DEFINED ENV{XILINX_XRT} AND DEFINED ENV{LD_LIBRARY_PATH}) AND DEFINED ENV{PYTHONPATH}))
        message(FATAL_ERROR "Environment has not been setup correctly!\n"
                            "XILINX_XRT and/or LD_LIBRARY_PATH and/or PYTHONPATH environment variable(s) are not set.\n"
                            "For further information see: https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/settingupvitisenvironment.html\n"
                            "Did you forget to run \"source /opt/xilinx/xrt/setup.sh\"?")
    endif()
else()
    message(STATUS "Skipping environment variable checks...")
endif()

# Set Connectivity Property
if(MP_EMBEDDED)
    set(CONNECTIVITY_SP "HP")
    set(MP_EMBEDDED_BOOL "true")
else()
    set(CONNECTIVITY_SP "bank")
    set(MP_EMBEDDED_BOOL "false")
endif()

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

if(MP_EMBEDDED)
    set(HOST_COMPILER "aarch64-linux-gnu-g++")
else()
    set(HOST_COMPILER "g++")
endif()

set(VITIS_COMPILER "$ENV{XILINX_VITIS}/bin/v++")

if("${MP_KERNEL}" STREQUAL "Stream2D")
    # Generate Completely Unrolled Loop Code
    MATH(EXPR nTiles "((${MP_SIZE_N}-${MP_SIZE_M}+${MP_SIZE_T}) / ${MP_SIZE_T})")
    function(index2D x y)
        MATH(EXPR index "(${nTiles} * (${nTiles} - 1)) / 2 - ((${nTiles} - ${y}) * (${nTiles} - ${y} - 1)) / 2 + ${x}")
        set(index ${index} PARENT_SCOPE)
    endfunction()

    MATH(EXPR nTiles_1 "${nTiles} - 1")
    foreach(y RANGE "${nTiles_1}")
        index2D(${y} ${y})
        MATH(EXPR y_1 "${y} + 1")
        set(Stream2DInitSource "${Stream2DInitSource}ScatterLaneStreamingUnit(${y}, ${y}, sT[${y}], sMu[${y}], sDf[${y}], sDg[${y}], sInv[${y}], Ti[${index}], Tj[${index}], 
                        mui[${index}], muj[${index}], dfi[${index}], dfj[${index}], dgi[${index}], dgj[${index}], invi[${index}], invj[${index}], 
                        rowAggregate[${index}], columnAggregate[${index}], sT[${y_1}], sMu[${y_1}], sDf[${y_1}], sDg[${y_1}], sInv[${y_1}]);\n\n")
        if(NOT (${nTiles_1} LESS ${y_1}))
            foreach(x RANGE ${y_1} ${nTiles_1})
                index2D(${x} ${y})
                MATH(EXPR index_1 "${index} - 1")
                set(Stream2DInitSource "${Stream2DInitSource}RowLaneStreamingUnit(${y}, ${x}, Ti[${index_1}], Tj[${index_1}], mui[${index_1}], muj[${index_1}],
                            dfi[${index_1}], dfj[${index_1}], dgi[${index_1}], dgj[${index_1}], invi[${index_1}],
                            invj[${index_1}], rowAggregate[${index_1}], columnAggregate[${index_1}], Ti[${index}],
                            Tj[${index}], mui[${index}], muj[${index}], dfi[${index}], dfj[${index}], dgi[${index}], dgj[${index}], 
                            invi[${index}], invj[${index}], rowAggregate[${index}], columnAggregate[${index}]);\n\n")
            endforeach()
        endif()

        index2D(${nTiles_1} ${y})
        set(Stream2DInitSource "${Stream2DInitSource}RowReductionUnit(${y}, rRow[${y}], rCol[${y}], rowAggregate[${index}], columnAggregate[${index}], rRow[${y_1}], rCol[${y_1}]);\n\n")
    endforeach()
    # Write Generated Code to File (will be included in the Stream2D)
    configure_file(include/kernel/Stream2DInit.hpp.in kernel/Stream2DInit.hpp)
endif()

# Host Application (Compile & Link)
if(MP_EMBEDDED)
    set(HOST_COMPILE_COMMAND
        -std=c++11 -DVITIS_PLATFORM=${MP_PLATFORM} -D__USE_XOPEN2K8
        -I"${CMAKE_CURRENT_BINARY_DIR}" -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        -I"$ENV{XILINX_VIVADO}/include/" -I${MP_SYSROOT}/usr/include/xrt/
        --sysroot=${MP_SYSROOT} -O3 -c)
    set(HOST_LINK_COMMAND -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++ -L${MP_SYSROOT}/usr/lib/ --sysroot=${MP_SYSROOT})
else()
    set(HOST_COMPILE_COMMAND
        -std=c++11 -DVITIS_PLATFORM=${MP_PLATFORM} -D__USE_XOPEN2K8
        -I"${CMAKE_CURRENT_BINARY_DIR}" -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        -I"$ENV{XILINX_VIVADO}/include/" -I"$ENV{XILINX_XRT}/include/"
        -O3 -c)
    set(HOST_LINK_COMMAND -L$ENV{XILINX_XRT}/lib/ -lOpenCL -lpthread -lrt -lstdc++)
endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o
                   COMMENT "Compiling Host Application (n = ${MP_SIZE_N}, m = ${MP_SIZE_M})"
                   COMMAND ${HOST_COMPILER} ${HOST_COMPILE_COMMAND} -o "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o" "${CMAKE_CURRENT_SOURCE_DIR}/host/MatrixProfileHost.cpp"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/host/MatrixProfileHost.cpp")

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost
                   COMMENT "Linking Host Application"
                   COMMAND ${HOST_COMPILER} -o "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost" "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o" ${HOST_LINK_COMMAND}
                   DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost.o")

add_custom_target(host DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost)

if(MP_QUICK_BUILD)
    message(WARNING "MP_QUICK_BUILD is set! This can reduce kernel performance, and increase the resources used by the kernel!")
    set(OPTIMIZATION_PARAMETER "-Oquick")
else()
    set(OPTIMIZATION_PARAMETER "-O3")
endif()

# Kernel (Compile & Link)
set(KERNEL_COMPILE_COMMAND
    --compile 
    -I"${CMAKE_CURRENT_BINARY_DIR}" -I"${CMAKE_CURRENT_SOURCE_DIR}/kernel"
    --config "${CMAKE_CURRENT_BINARY_DIR}/config/common-config.cfg"
    --config "${CMAKE_CURRENT_BINARY_DIR}/config/compile-config.cfg"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include")

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo
                   COMMENT "Compiling ${MP_KERNEL}-Kernel (n = ${MP_SIZE_N}, m = ${MP_SIZE_M}, target = ${MP_TARGET})"
                   COMMAND ${CMAKE_COMMAND} -E env "XCL_EMULATION_MODE=${MP_TARGET}" "LIBRARY_PATH=${MP_LIBRARY_PATH}" ${VITIS_COMPILER} --target ${MP_TARGET} ${KERNEL_COMPILE_COMMAND} ${OPTIMIZATION_PARAMETER} -o"${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo" "${CMAKE_CURRENT_SOURCE_DIR}/kernel/MatrixProfileKernel${MP_KERNEL}.cpp"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/MatrixProfileKernel${MP_KERNEL}.cpp"
                           "${CMAKE_CURRENT_BINARY_DIR}/config/common-config.cfg"
                           "${CMAKE_CURRENT_BINARY_DIR}/config/compile-config.cfg")

add_custom_target(compile DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo)

set(KERNEL_LINK_COMMAND
    --link --config "${CMAKE_CURRENT_BINARY_DIR}/config/common-config.cfg" --config "${CMAKE_CURRENT_BINARY_DIR}/config/link-config.cfg")


if(("${MP_TARGET}" STREQUAL "hw_emu") OR ("${MP_TARGET}" STREQUAL "sw_emu"))
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/emconfig.json"
                    COMMENT "Generating emconfig.json file for emulation."
                    COMMAND $ENV{XILINX_VITIS}/bin/emconfigutil --platform ${MP_PLATFORM})
endif()

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernel.xclbin"
                   COMMENT "Linking ${MP_KERNEL}-Kernel (n = ${MP_SIZE_N}, m = ${MP_SIZE_M}, target = ${MP_TARGET})"
                   COMMAND ${CMAKE_COMMAND} -E env "XCL_EMULATION_MODE=${MP_TARGET}" "LIBRARY_PATH=${MP_LIBRARY_PATH}" ${VITIS_COMPILER} --target ${MP_TARGET} ${KERNEL_LINK_COMMAND} ${OPTIMIZATION_PARAMETER} -o"${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernel.xclbin" "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo"
                   DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernelTLF.xo"
                           "${CMAKE_CURRENT_BINARY_DIR}/config/common-config.cfg"
                           "${CMAKE_CURRENT_BINARY_DIR}/config/link-config.cfg")

if(("${MP_TARGET}" STREQUAL "hw_emu") OR ("${MP_TARGET}" STREQUAL "sw_emu"))
    add_custom_target(link DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/emconfig.json" "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernel.xclbin")
else()
    add_custom_target(link DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernel.xclbin")
endif()

# Packaging (SD-Card) Only Relevant for Embedded Platforms
if(MP_EMBEDDED)
    set(PACKAGE_COMMAND --package --config "${CMAKE_CURRENT_BINARY_DIR}/config/package-config.cfg" --package.sd_dir "${CMAKE_CURRENT_SOURCE_DIR}/data")

    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/package/sd_card.img" "${CMAKE_CURRENT_BINARY_DIR}/package/launch_hw_emu.sh"
                       COMMENT "Packaging SD-Card for Embedded Platforms"
                       COMMAND ${CMAKE_COMMAND} -E env "XCL_EMULATION_MODE=${MP_TARGET}" "LIBRARY_PATH=${MP_LIBRARY_PATH}" ${VITIS_COMPILER} --target ${MP_TARGET} ${PACKAGE_COMMAND} "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernel.xclbin"
                       DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileHost"
                               "${CMAKE_CURRENT_BINARY_DIR}/MatrixProfileKernel.xclbin"
                               "${CMAKE_CURRENT_BINARY_DIR}/config/package-config.cfg")

    add_custom_target(package_sd DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/package/sd_card.img" "${CMAKE_CURRENT_BINARY_DIR}/package/launch_hw_emu.sh")
    
    # Launch the Emulator via the generated script and set necessary environment variables
    add_custom_target(launch_emulator COMMAND ${CMAKE_COMMAND} -E env "XCL_EMULATION_MODE=${MP_TARGET}" "LIBRARY_PATH=${MP_LIBRARY_PATH}" /bin/bash "${CMAKE_CURRENT_BINARY_DIR}/package/launch_hw_emu.sh" 
                                      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/package/sd_card.img" "${CMAKE_CURRENT_BINARY_DIR}/package/launch_hw_emu.sh")
endif()

# Software Testing via GoogleTest (https://github.com/google/googletest)
if(BUILD_TESTS)
    if(NOT(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/CMakeLists.txt"))
        message(FATAL_ERROR "Attempting to build tests without a valid installation of GoogleTest.\n"
                            "Make sure to either clone the repository with '--recursive' or download it after cloning via:\n"
                            "'git submodule update --init'")
    endif()
    
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
