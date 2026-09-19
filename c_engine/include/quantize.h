#ifndef EI_QUANTIZE_H
#define EI_QUANTIZE_H

#include <stdint.h>

#include "tensor.h"

// one int8 layer: per-channel symmetric weights, int32 bias at scale
// in_scale * w_scale[c], and a q31 multiplier + shift per output channel
// that maps the int32 accumulator onto the output scale
typedef struct {
    const int8_t *w;
    const int32_t *bias;
    const int32_t *mult;
    const int8_t *shift;
    int32_t in_zp, out_zp;
    // relu folds in as act_min = out_zp
    int32_t act_min, act_max;
    float out_scale;
} layer_q_t;

// saturating rounding doubling high mul, (a * b * 2) >> 32 with rounding
static inline int32_t srdhm(int32_t a, int32_t b)
{
    if (a == b && a == INT32_MIN)
        return INT32_MAX;
    int64_t ab = (int64_t)a * b;
    int64_t nudge = ab >= 0 ? (1 << 30) : 1 - (1 << 30);
    return (int32_t)((ab + nudge) / ((int64_t)1 << 31));
}

// rounding divide by 2^exp, ties away from zero
static inline int32_t rdbpot(int32_t x, int exp)
{
    int32_t mask = (int32_t)(((int64_t)1 << exp) - 1);
    int32_t threshold = (mask >> 1) + (x < 0);
    return (x >> exp) + ((x & mask) > threshold);
}

// acc * real multiplier, where real = mult * 2^(shift - 31)
static inline int32_t requant(int32_t acc, int32_t mult, int shift)
{
    int left = shift > 0 ? shift : 0;
    int right = shift > 0 ? 0 : -shift;
    return rdbpot(srdhm(acc * (1 << left), mult), right);
}

static inline int8_t clamp_s8(int32_t v, int32_t lo, int32_t hi)
{
    return (int8_t)(v < lo ? lo : v > hi ? hi : v);
}

// float -> int8 with out's scale and zero point, ties away from zero
void quantize_tensor(const float *x, tensor_t *out);
void dequantize_tensor(const tensor_t *in, float *out);

// out[i][j] = requant(bias[j] + sum_t (a[i][t] - in_zp) * w[j][t]), clamped.
// a is m x k, w is n x k, both row-major: pointwise convs and dense layers
void qmatmul_s8(const int8_t *a, const layer_q_t *l, int8_t *out, int m, int n, int k);

#endif
