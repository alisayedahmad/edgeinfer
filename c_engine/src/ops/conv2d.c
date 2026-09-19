#include "ops.h"

// direct convolution, no im2col. for each output pixel and channel, walk
// the kernel window and dot the input pixel's channels against the weights.
// padded taps are skipped, which is the same as multiplying by zero
void conv2d_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p)
{
    const float *x = in->data;
    float *y = out->data;
    for (int oy = 0; oy < out->h; oy++)
        for (int ox = 0; ox < out->w; ox++)
            for (int co = 0; co < out->c; co++) {
                float acc = b ? b[co] : 0.0f;
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
                y[(oy * out->w + ox) * out->c + co] = acc;
            }
}

// same loop in int8. 1x1 stride-1 convs are a matmul over pixels, so they
// go straight to qmatmul_s8
void conv2d_s8(const tensor_t *in, tensor_t *out, const layer_q_t *l, const conv_params_t *p)
{
    if (p->kh == 1 && p->kw == 1 && p->sh == 1 && p->sw == 1 && p->ph == 0 && p->pw == 0) {
        qmatmul_s8(in->data, l, out->data, in->h * in->w, out->c, in->c);
        return;
    }
    const int8_t *x = in->data;
    int8_t *y = out->data;
    for (int oy = 0; oy < out->h; oy++)
        for (int ox = 0; ox < out->w; ox++)
            for (int co = 0; co < out->c; co++) {
                int32_t acc = l->bias[co];
                for (int ky = 0; ky < p->kh; ky++) {
                    int iy = oy * p->sh - p->ph + ky;
                    if (iy < 0 || iy >= in->h)
                        continue;
                    for (int kx = 0; kx < p->kw; kx++) {
                        int ix = ox * p->sw - p->pw + kx;
                        if (ix < 0 || ix >= in->w)
                            continue;
                        const int8_t *xp = x + (iy * in->w + ix) * in->c;
                        const int8_t *wp = l->w + ((co * p->kh + ky) * p->kw + kx) * in->c;
                        for (int ci = 0; ci < in->c; ci++)
                            acc += (xp[ci] - l->in_zp) * wp[ci];
                    }
                }
                int32_t v = requant(acc, l->mult[co], l->shift[co]) + l->out_zp;
                y[(oy * out->w + ox) * out->c + co] = clamp_s8(v, l->act_min, l->act_max);
            }
}
