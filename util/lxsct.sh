#!/bin/bash
echo "connect -url tcp:localhost:1440
tfile copy -to-host /media/sd-mmcblk0p1/$1 $1" | xsct
