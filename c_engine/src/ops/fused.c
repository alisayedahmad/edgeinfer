#include "ops.h"

// conv + bn + relu as one pass. bn was folded into w and b at export time
// (w' = w * gamma / sqrt(var + eps), b' = beta - mean * gamma / sqrt(var + eps)),
// so bn costs nothing here, and relu is a clamp on the single output write.
// the inner loops are the same as conv2d_f32 and depthwise_f32 so fused vs
// unfused timing isolates the fusion itself

void conv2d_bn_relu_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p)
{
    const float *x = in->data;
    float *y = out->data;
    for (int oy = 0; oy < out->h; oy++)
        for (int ox = 0; ox < out->w; ox++)
            for (int co = 0; co < out->c; co++) {
                float acc = b[co];
                for (int ky = 0; ky < p->kh; ky++) {
                    int iy = oy * p->sh - p->ph + ky;
                    if (iy < 0 || iy >= in->h)
                        continue;
                    for (int kx = 0; kx < p->kw; kx++) {
                        int ix = ox * p->sw - p->pw + kx;
                        if (ix < 0 || ix >= in->w)
                            continue;
                        const float *xp = x + (iy * in->w + ix) * in->c;
                        const float *wp = w + ((co * p->kh + ky) * p->kw + kx) * in->c;
                        for (int ci = 0; ci < in->c; ci++)
                            acc += xp[ci] * wp[ci];
                    }
                }
                y[(oy * out->w + ox) * out->c + co] = acc > 0.0f ? acc : 0.0f;
            }
}

// relu clamps each pixel's channels while they are still in cache
void depthwise_bn_relu_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p)
{
    const float *x = in->data;
    int c = in->c;
    for (int oy = 0; oy < out->h; oy++)
        for (int ox = 0; ox < out->w; ox++) {
            float *yp = (float *)out->data + (oy * out->w + ox) * c;
            for (int ch = 0; ch < c; ch++)
                yp[ch] = b[ch];
            for (int ky = 0; ky < p->kh; ky++) {
                int iy = oy * p->sh - p->ph + ky;
                if (iy < 0 || iy >= in->h)
                    continue;
                for (int kx = 0; kx < p->kw; kx++) {
                    int ix = ox * p->sw - p->pw + kx;
                    if (ix < 0 || ix >= in->w)
                        continue;
                    const float *xp = x + (iy * in->w + ix) * c;
                    const float *wp = w + (ky * p->kw + kx) * c;
                    for (int ch = 0; ch < c; ch++)
                        yp[ch] += xp[ch] * wp[ch];
                }
            }
            for (int ch = 0; ch < c; ch++)
                yp[ch] = yp[ch] > 0.0f ? yp[ch] : 0.0f;
        }
}
