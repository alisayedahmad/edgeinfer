#include "ops.h"

// fully connected on a flattened input, one dot product per output
void dense_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b)
{
    const float *x = in->data;
    float *y = out->data;
    int n = tensor_len(in);
    for (int j = 0; j < out->c; j++) {
        const float *wp = w + j * n;
        float acc = b[j];
        for (int i = 0; i < n; i++)
            acc += x[i] * wp[i];
        y[j] = acc;
    }
}

// a 1-row quantized matmul
void dense_s8(const tensor_t *in, tensor_t *out, const layer_q_t *l)
{
    qmatmul_s8(in->data, l, out->data, 1, out->c, tensor_len(in));
}
