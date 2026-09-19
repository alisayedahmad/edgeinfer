#include "ops.h"

// inference-mode batch norm as a per-channel affine: y = x * scale + shift,
// scale = gamma / sqrt(var + eps), shift = beta - mean * scale
void batchnorm_f32(const tensor_t *in, tensor_t *out, const float *scale, const float *shift)
{
    const float *x = in->data;
    float *y = out->data;
    int c = in->c, n = in->h * in->w;
    for (int i = 0; i < n; i++)
        for (int ch = 0; ch < c; ch++)
            y[i * c + ch] = x[i * c + ch] * scale[ch] + shift[ch];
}

void relu_f32(const tensor_t *in, tensor_t *out)
{
    const float *x = in->data;
    float *y = out->data;
    int n = tensor_len(in);
    for (int i = 0; i < n; i++)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
