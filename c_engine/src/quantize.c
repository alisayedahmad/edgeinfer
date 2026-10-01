#include <math.h>

#include "dsp.h"
#include "quantize.h"

// taps per row the unpack buffer holds, 512 bytes of stack
#define QMAT_ROW 256

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
// accumulator to output byte, on channel j's multiplier
static inline int8_t requant_s8(int32_t acc, const layer_q_t *l, int j)
{
    int32_t v = requant(acc, l->mult[j], l->shift[j]) + l->out_zp;
    return clamp_s8(v, l->act_min, l->act_max);
}

void qmatmul_s8(const int8_t *a, const layer_q_t *l, int8_t *out, int m, int n, int k)
{
    uint32_t zp2 = ((uint32_t)l->in_zp << 16) | ((uint32_t)l->in_zp & 0xffffu);
    // every output column reads the same activations, so unpack the row once
    // instead of n times. a wider row than this falls through to the scalar
    // tail, which is correct but slow, and no layer here comes close
    uint32_t arow[QMAT_ROW / 2];
    for (int i = 0; i < m; i++) {
        const int8_t *ap = a + i * k;
        int blocks = k <= QMAT_ROW ? k / 4 : 0;
        for (int b = 0; b < blocks; b++) {
            uint32_t av = read32(ap + 4 * b);
            // sxtb16 takes bytes 0 and 2, the rotated one takes 1 and 3, so
            // the weights unpack into the same pairing further down
            arow[2 * b] = ssub16(sxtb16(av), zp2);
            arow[2 * b + 1] = ssub16(sxtb16_ror8(av), zp2);
        }
        int j = 0;
        // two output columns per pass, so each unpacked activation pair is
        // loaded once and feeds two accumulators
        for (; j <= n - 2; j += 2) {
            const int8_t *w0 = l->w + j * k, *w1 = w0 + k;
            int32_t acc0 = l->bias[j], acc1 = l->bias[j + 1];
            int t = 0;
            for (int b = 0; b < blocks; b++, t += 4) {
                uint32_t lo = arow[2 * b], hi = arow[2 * b + 1];
                uint32_t v0 = read32(w0 + t), v1 = read32(w1 + t);
                acc0 = smlad(lo, sxtb16(v0), acc0);
                acc0 = smlad(hi, sxtb16_ror8(v0), acc0);
                acc1 = smlad(lo, sxtb16(v1), acc1);
                acc1 = smlad(hi, sxtb16_ror8(v1), acc1);
            }
            for (; t < k; t++) {
                acc0 += (ap[t] - l->in_zp) * w0[t];
                acc1 += (ap[t] - l->in_zp) * w1[t];
            }
            out[i * n + j] = requant_s8(acc0, l, j);
            out[i * n + j + 1] = requant_s8(acc1, l, j + 1);
        }
        for (; j < n; j++) {
            const int8_t *wp = l->w + j * k;
            int32_t acc = l->bias[j];
            int t = 0;
            for (int b = 0; b < blocks; b++, t += 4) {
                uint32_t wv = read32(wp + t);
                acc = smlad(arow[2 * b], sxtb16(wv), acc);
                acc = smlad(arow[2 * b + 1], sxtb16_ror8(wv), acc);
            }
            for (; t < k; t++)
                acc += (ap[t] - l->in_zp) * wp[t];
            out[i * n + j] = requant_s8(acc, l, j);
        }
    }
}
