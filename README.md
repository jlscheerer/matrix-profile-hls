# Matrix Profile Computation on Xilinx FPGAs [![BSD3 License](https://img.shields.io/badge/License-BSDv3-blue.svg)](LICENSE.md) [![Build Status](https://travis-ci.com/jlscheerer/matrix-profile-hls.svg?token=dmssrYE2KgMinUZ9Pecp&branch=master)](https://travis-ci.com/jlscheerer/matrix-profile-hls)

This repository includes multiple pure Vitis™ HLS implementations of the Matrix Profile Computation Algorithm [SCAMP](https://github.com/zpzim/SCAMP) for Xilinx FPGAs, using Xilinx Vitis™ to instantiate memory and PCIe controllers and interface with the host.

The Matrix Profile is a novel data structure with corresponding algorithms (stomp, regimes, motifs, etc.) developed by the [Keogh](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) and [Mueen](https://www.cs.unm.edu/~mueen/) research groups at UC-Riverside and the University of New Mexico. 

The source files for the different implementation of the compute kernel can be found under [``kernel/MatrixProfileKernelStreamless.cpp``](kernel/MatrixProfileKernelStreamless.cpp), [``kernel/MatrixProfileKernelStream1D.cpp``](kernel/MatrixProfileKernelStream1D.cpp) and [``kernel/MatrixProfileKernelStream2D.cpp``](kernel/MatrixProfileKernelStream2D.cpp).

The host application is in [``host/MatrixProfileHost.cpp``](host/MatrixProfileHost.cpp). This repository contains a light-weight OpenCL™ wrapper for the interaction with the FPGA kernel, which is located in [``include/host/OpenCL.hpp``](include/host/OpenCL.hpp).

## Getting Started
### Cloning the repository
This project uses Google's open source testing and mocking framework [GoogleTest](https://github.com/google/googletest) to test the different kernels in software.

Since GoogleTest is included as a submodule, make sure to to clone the repository with ``--recursive`` if you plan on running the (software) tests. If the repository was cloned non-recursively previously, use ``git submodule update --init`` to clone the required submodule (GoogleTest).

### Prerequisites
To build and run the kernels in hardware (simulation) [Xilinx Vitis](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2020-2.html) must be installed and the corresponding ``PATH``-variables must be set. In particular, this typically requires [setting up the environment to run the Vitis™ software platform](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/settingupvitisenvironment.html):
```bash
source /opt/Xilinx/Vitis/2020.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
```
and setting the correct ``XCL_EMULATION_MODE`` in case of software (``sw_emu``) or hardware (``hw_emu``) emulation:
```bash
export XCL_EMULATION_MODE=hw_emu
```
## Build and Run
### Configuration and Building
This project is configured and built using CMake. Most parameters must be set at configuration-time, as they are used to specialize the hardware.

An example of configuring the kernel (starting from the ``root`` directory):

```bash
mkdir build && cd build
cmake .. -DMP_KERNEL=Stream1D -DMP_TARGET=hw -DMP_DATA_TYPE=double -DMP_SIZE_N=256 -DMP_SIZE_M=32 -DMP_SIZE_T=64
make host
make compile
make link
```

### Build Parameters

| **CMake Parameter** | **Description**           | **Values**                                                            |
|---------------------|---------------------------|-----------------------------------------------------------------------|
| ``MP_KERNEL``       | Kernel-Implementation     | ``Streamless``, ``Stream1D``, ``Stream2D``                            |
| ``MP_DATA_TYPE``    | Data Type for Computation | ``double``, ``float``, ``ap16_t``, ``ap24_t``, ``ap32_t``, ``ap64_t`` |
| ``MP_TARGET``       | Compilation Target        | ``sw_emu``, ``hw_emu``, ``hw``                                        |
| ``DMP_SIZE_N``      | Length of the Time Series |                                                                       |
| ``DMP_SIZE_M``      | Subsequence Length        |                                                                       |
| ``DMP_SIZE_T``      | Tile-Size                 | *only applicable for Stream1D/Stream2D-Kernel*                        |

For a more comprehensive list of parameters (e.g., targeting ``EMBEDDED``-Platforms) see [``CMakeLists.txt``](CMakeLists.txt).

Per default the build targets the Alveo U250 acceleration board, but this can be configured using the ``MP_PLATFORM`` CMake parameter.

### Executing the Kernel
After having ``build`` the host application and `link`ed the Kernel, execute the Kernel on input [``data/binary/small128_syn.tsb``](data/binary/small128_syn.tsb) (run this in the ``build`` directory) via:
```bash
./MatrixProfileHost -b MatrixProfileKernel.xclbin -i ../data/binary/small128_syn.tsb --verbose
```
A list of example datasets as well as instruction on how to use your own dataset can be found [here](data/).

## Testing
This project uses Google's open source testing and mocking framework [GoogleTest](https://github.com/google/googletest) to test the different kernels in software.

To build and run the test executables:
```bash
mkdir build && cd build
cmake .. -DSKIP_CHECKS=ON -DBUILD_TESTS=ON
make && make test
```

The corresponding source files can be found under [``test/TestStreamlessKernel.cpp``](test/TestStreamlessKernel.cpp), [``test/TestStream1DKernel.cpp``](test/TestStream1DKernel.cpp) and [``test/TestStream2DKernel.cpp``](test/TestStream2DKernel.cpp)

## Bugs
If you experience bugs, or have suggestions for improvements, please use the issue tracker to report them.
