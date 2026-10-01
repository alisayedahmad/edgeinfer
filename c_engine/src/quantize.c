#include <math.h>
#include <string.h>

#include "quantize.h"

void quantize_tensor(const float *x, tensor_t *out)
{
    int8_t *q = out->data;
    int n = tensor_len(out);
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)roundf(x[i] / out->scale) + out->zero_point;
        q[i] = clamp_s8(v, -128, 127);
    }
}

void dequantize_tensor(const tensor_t *in, float *out)
{
    const int8_t *q = in->data;
    int n = tensor_len(in);
    for (int i = 0; i < n; i++)
        out[i] = (float)(q[i] - in->zero_point) * in->scale;
}

// int8 x int8 -> int32 dot products, then one requant per output. both
// operands are walked along contiguous rows, which is why activations are
// hwc and weights [out][in]
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

void qmatmul_s8(const int8_t *a, const layer_q_t *l, int8_t *out, int m, int n, int k)
{
    uint32_t zp2 = ((uint32_t)l->in_zp << 16) | ((uint32_t)l->in_zp & 0xffffu);
    for (int i = 0; i < m; i++) {
        const int8_t *ap = a + i * k;
        for (int j = 0; j < n; j++) {
            const int8_t *wp = l->w + j * k;
            int32_t acc = l->bias[j];
            int t = 0;
            // four taps per pass: one load each side, then two smlads. sxtb16
            // takes bytes 0 and 2, the rotated one takes 1 and 3, so both
            // sides are paired the same way and the four products all land
            for (; t <= k - 4; t += 4) {
                uint32_t av = read32(ap + t), wv = read32(wp + t);
                acc = smlad(ssub16(sxtb16(av), zp2), sxtb16(wv), acc);
                acc = smlad(ssub16(sxtb16_ror8(av), zp2), sxtb16_ror8(wv), acc);
            }
            for (; t < k; t++)
                acc += (ap[t] - l->in_zp) * wp[t];
            int32_t v = requant(acc, l->mult[j], l->shift[j]) + l->out_zp;
            out[i * n + j] = clamp_s8(v, l->act_min, l->act_max);
        }
    }
}
