#include "dsp.h"
#include "ops.h"

// direct convolution, no im2col. for each output pixel and channel, walk
// the kernel window and dot the input pixel's channels against the weights.
// padded taps are skipped, which is the same as multiplying by zero.
// the dot product runs over cin, so the first layer (cin = 1) gets no
// vectorization and costs far more per mac than the pointwise convs
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

// first layer: one input channel, so a kernel row is contiguous on both sides.
// the padding is clipped once per output pixel instead of tested per tap, and
// the row pointers walk forward, which is where this layer's cost actually was
static void conv2d_s8_c1(const tensor_t *in, tensor_t *out, const layer_q_t *l, const conv_params_t *p)
{
    const int8_t *x = in->data;
    int8_t *y = out->data;
    uint32_t zp2 = ((uint32_t)l->in_zp << 16) | ((uint32_t)l->in_zp & 0xffffu);
    int row = p->kh * p->kw;
    for (int oy = 0; oy < out->h; oy++) {
        int top = oy * p->sh - p->ph;
        int ky0 = top < 0 ? -top : 0;
        int ky1 = top + p->kh > in->h ? in->h - top : p->kh;
        for (int ox = 0; ox < out->w; ox++) {
            int ix0 = ox * p->sw - p->pw;
            int kx0 = ix0 < 0 ? -ix0 : 0;
            int kx1 = ix0 + p->kw > in->w ? in->w - ix0 : p->kw;
            for (int co = 0; co < out->c; co++) {
                int32_t acc = l->bias[co];
                const int8_t *xp = x + (top + ky0) * in->w + ix0;
                const int8_t *wp = l->w + co * row + ky0 * p->kw;
                for (int ky = ky0; ky < ky1; ky++, xp += in->w, wp += p->kw) {
                    int kx = kx0;
                    for (; kx <= kx1 - 4; kx += 4) {
                        uint32_t av = read32(xp + kx), wv = read32(wp + kx);
                        acc = smlad(ssub16(sxtb16(av), zp2), sxtb16(wv), acc);
                        acc = smlad(ssub16(sxtb16_ror8(av), zp2), sxtb16_ror8(wv), acc);
                    }
                    for (; kx < kx1; kx++)
                        acc += (xp[kx] - l->in_zp) * wp[kx];
                }
                int32_t v = requant(acc, l->mult[co], l->shift[co]) + l->out_zp;
                y[(oy * out->w + ox) * out->c + co] = clamp_s8(v, l->act_min, l->act_max);
            }
        }
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
    if (in->c == 1) {
        conv2d_s8_c1(in, out, l, p);
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
