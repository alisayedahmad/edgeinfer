#include "ops.h"

// channels processed per pass in int8, bounds the stack accumulator
#define DW_BLOCK 64

// depthwise conv: one kh x kw filter per channel, no mixing across channels.
// channels are the inner loop, contiguous in both hwc input and [kh][kw][c]
// weights, so each tap is a vector multiply-add straight into the output
void depthwise_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p)
{
    const float *x = in->data;
    int c = in->c;
    for (int oy = 0; oy < out->h; oy++)
        for (int ox = 0; ox < out->w; ox++) {
            float *yp = (float *)out->data + (oy * out->w + ox) * c;
            for (int ch = 0; ch < c; ch++)
                yp[ch] = b ? b[ch] : 0.0f;
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
        }
}

// int8 can't accumulate in the output, so channels go in blocks through a
// small int32 accumulator on the stack
void depthwise_s8(const tensor_t *in, tensor_t *out, const layer_q_t *l, const conv_params_t *p)
{
    const int8_t *x = in->data;
    int c = in->c;
    int32_t acc[DW_BLOCK];
    for (int oy = 0; oy < out->h; oy++)
        for (int ox = 0; ox < out->w; ox++)
            for (int c0 = 0; c0 < c; c0 += DW_BLOCK) {
                int n = c - c0 < DW_BLOCK ? c - c0 : DW_BLOCK;
                for (int i = 0; i < n; i++)
                    acc[i] = l->bias[c0 + i];
                for (int ky = 0; ky < p->kh; ky++) {
                    int iy = oy * p->sh - p->ph + ky;
                    if (iy < 0 || iy >= in->h)
                        continue;
                    for (int kx = 0; kx < p->kw; kx++) {
                        int ix = ox * p->sw - p->pw + kx;
                        if (ix < 0 || ix >= in->w)
                            continue;
                        const int8_t *xp = x + (iy * in->w + ix) * c + c0;
                        const int8_t *wp = l->w + (ky * p->kw + kx) * c + c0;
                        for (int i = 0; i < n; i++)
                            acc[i] += (xp[i] - l->in_zp) * wp[i];
                    }
                }
                int8_t *yp = (int8_t *)out->data + (oy * out->w + ox) * c + c0;
                for (int i = 0; i < n; i++) {
                    int32_t v = requant(acc[i], l->mult[c0 + i], l->shift[c0 + i]) + l->out_zp;
                    yp[i] = clamp_s8(v, l->act_min, l->act_max);
                }
            }
}
