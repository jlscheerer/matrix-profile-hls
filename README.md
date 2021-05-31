# Matrix Profile Computation on FPGA

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
In the /build directory run:
```bash
./package/launch_hw_emu.sh
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
./matrix-profile-hls -b mp_binary_container.xclbin -i small8_syn --verbose
```
