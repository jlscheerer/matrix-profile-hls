# Matrix Profile Computation on FPGA

[![Build Status](https://travis-ci.com/jlscheerer/matrix-profile-hls.svg?token=dmssrYE2KgMinUZ9Pecp&branch=master)](https://travis-ci.com/jlscheerer/matrix-profile-hls)
[![Xilinx Vitis](https://img.shields.io/badge/Powered%20by-Xilinx%20Vitis-orange.svg)](https://www.xilinx.com/products/design-tools/vitis/vitis-platform.html)
[![BSD3 License](https://img.shields.io/badge/License-BSDv3-blue.svg)](LICENSE.md)

### Configuration and Building
```bash
mkdir build && cd build
cmake ..
make host
make compile
make link
make package_sd
```

### Launch the Emulator
```bash
make launch_emulator
```

### Configure Environment Variables for Emulator
```bash
mount /dev/mmcblk0p1 /mnt
cd /mnt
export LD_LIBRARY_PATH=/mnt:/tmp:$LD_LIBRARY_PATH
export XCL_EMULATION_MODE=hw_emu
export XILINX_XRT=/usr
export XILINX_VITIS=/mnt
```
> Source: https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/runemulation1.html

### Running
```bash
./MatrixProfileHost -b MatrixProfileKernel.xclbin -i small8_syn --verbose
```

> Data-Set Source: https://github.com/matrix-profile-foundation/mpf-datasets