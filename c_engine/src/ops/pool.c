#include "ops.h"

// global average pool, h x w x c -> 1 x 1 x c
void avgpool_f32(const tensor_t *in, tensor_t *out)
{
    const float *x = in->data;
    float *y = out->data;
    int n = in->h * in->w;
    for (int ch = 0; ch < in->c; ch++)
        y[ch] = 0.0f;
    for (int i = 0; i < n; i++)
        for (int ch = 0; ch < in->c; ch++)
            y[ch] += x[i * in->c + ch];
    for (int ch = 0; ch < in->c; ch++)
        y[ch] /= (float)n;
}

// averages raw int8 values, so the output keeps the input's scale and zero
// point. rounds half away from zero like tflite's reference kernel
void avgpool_s8(const tensor_t *in, tensor_t *out)
{
    const int8_t *x = in->data;
    int8_t *y = out->data;
    int n = in->h * in->w;
    for (int ch = 0; ch < in->c; ch++) {
        int32_t acc = 0;
        for (int i = 0; i < n; i++)
            acc += x[i * in->c + ch];
        acc = acc > 0 ? (acc + n / 2) / n : (acc - n / 2) / n;
        y[ch] = clamp_s8(acc, -128, 127);
    }
}
