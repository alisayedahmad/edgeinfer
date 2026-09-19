#!/bin/sh
# run the cortex-m4 build under qemu. -icount shift=0 makes every instruction
# take 1 ns of virtual time, so the 25 mhz systick advances one tick per 40
# instructions and per-op counts are deterministic. output is json over
# semihosting. needs qemu 5.2+ for the mps2-an386 (cortex-m4) machine
set -e
ELF=${1:-c_engine/build/arm/edgeinfer.elf}
exec qemu-system-arm \
    -M mps2-an386 -cpu cortex-m4 \
    -nographic -monitor none -serial none \
    -semihosting-config enable=on,target=native \
    -icount shift=0 \
    -kernel "$ELF"
