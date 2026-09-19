#include <math.h>

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
void qmatmul_s8(const int8_t *a, const layer_q_t *l, int8_t *out, int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        const int8_t *ap = a + i * k;
        for (int j = 0; j < n; j++) {
            const int8_t *wp = l->w + j * k;
            int32_t acc = l->bias[j];
            for (int t = 0; t < k; t++)
                acc += (ap[t] - l->in_zp) * wp[t];
            int32_t v = requant(acc, l->mult[j], l->shift[j]) + l->out_zp;
            out[i * n + j] = clamp_s8(v, l->act_min, l->act_max);
        }
    }
}
