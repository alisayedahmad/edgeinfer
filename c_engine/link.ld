/* cortex-m4 budget: 1 mb flash, 256 kb ram, matching the mps2-an386 qemu
   machine. weights are const so they stay in flash, the arena and the mfcc
   buffers land in .bss, and the link fails if they do not fit */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 1M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 256K
}

_stack_size = 8K;
_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } > FLASH

    .text :
    {
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.attributes) }
}

ASSERT(_ebss + _stack_size <= ORIGIN(RAM) + LENGTH(RAM), "out of ram: arena + bss + stack exceed 256k")
