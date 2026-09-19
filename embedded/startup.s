/* minimal cortex-m4 startup: vector table, .data copy, .bss zero, main */
    .syntax unified
    .cpu cortex-m4
    .thumb

    .section .isr_vector, "a"
    .word _estack
    .word Reset_Handler
    .word Default_Handler   /* nmi */
    .word Default_Handler   /* hardfault */
    .word Default_Handler   /* memmanage */
    .word Default_Handler   /* busfault */
    .word Default_Handler   /* usagefault */
    .word 0
    .word 0
    .word 0
    .word 0
    .word Default_Handler   /* svcall */
    .word Default_Handler   /* debugmon */
    .word 0
    .word Default_Handler   /* pendsv */
    .word SysTick_Handler

    .text
    .thumb_func
    .global Reset_Handler
Reset_Handler:
    ldr   r0, =_sidata
    ldr   r1, =_sdata
    ldr   r2, =_edata
copy_data:
    cmp   r1, r2
    bhs   zero_bss
    ldr   r3, [r0], #4
    str   r3, [r1], #4
    b     copy_data
zero_bss:
    ldr   r1, =_sbss
    ldr   r2, =_ebss
    movs  r3, #0
bss_loop:
    cmp   r1, r2
    bhs   call_main
    str   r3, [r1], #4
    b     bss_loop
call_main:
    bl    main
hang:
    b     hang

    .thumb_func
    .weak Default_Handler
Default_Handler:
    b     Default_Handler
