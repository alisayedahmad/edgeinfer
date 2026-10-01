#ifndef EI_DSP_H
#define EI_DSP_H

#include <stdint.h>
#include <string.h>

// whether a kernel should hand-pack four taps into sxtb16 and smlad. that is
// a win where those instructions exist. where they do not, the portable
// versions below are correct but slower than the plain loop they replace,
// because hand-packing is what stops a compiler vectorising it, so kernels
// with a plain alternative keep it. defining EI_PACKED forces the packed
// path on the host, which is how the operator tests reach it
#ifndef EI_PACKED
#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP
#define EI_PACKED 1
#else
#define EI_PACKED 0
#endif
#endif

// the cortex-m4 dsp instructions the int8 kernels run on, with portable
// versions so the same kernel builds and is tested on the host. sxtb16
// unpacks two of the four bytes into sign-extended halfwords, ssub16 takes
// the zero point off both at once, smlad does two macs in one instruction
#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP

static inline uint32_t sxtb16(uint32_t v)
{
    uint32_t r;
    __asm__("sxtb16 %0, %1" : "=r"(r) : "r"(v));
    return r;
}

static inline uint32_t sxtb16_ror8(uint32_t v)
{
    uint32_t r;
    __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(v));
    return r;
}

static inline uint32_t ssub16(uint32_t a, uint32_t b)
{
    uint32_t r;
    __asm__("ssub16 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    int32_t r;
    __asm__("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}

#else

static inline uint32_t sxtb16(uint32_t v)
{
    return ((uint32_t)(int32_t)(int8_t)v & 0xffffu) | ((uint32_t)(int32_t)(int8_t)(v >> 16) << 16);
}

static inline uint32_t sxtb16_ror8(uint32_t v)
{
    return sxtb16((v >> 8) | (v << 24));
}

static inline uint32_t ssub16(uint32_t a, uint32_t b)
{
    return (((a & 0xffffu) - (b & 0xffffu)) & 0xffffu) | (((a >> 16) - (b >> 16)) << 16);
}

static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    return acc + (int16_t)a * (int16_t)b + (int16_t)(a >> 16) * (int16_t)(b >> 16);
}

#endif

// one unaligned 32-bit load, which is a single ldr everywhere it matters
static inline uint32_t read32(const int8_t *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

#endif
