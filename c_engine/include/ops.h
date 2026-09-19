#ifndef EI_OPS_H
#define EI_OPS_H

#include "quantize.h"
#include "tensor.h"

typedef struct {
    int kh, kw, sh, sw, ph, pw;
} conv_params_t;

// weight layouts match hwc activations: conv [cout][kh][kw][cin],
// depthwise [kh][kw][c], dense [out][in]. bias may be null

// fp32, unfused: conv, then bn and relu as separate passes
void conv2d_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p);
void depthwise_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p);
void batchnorm_f32(const tensor_t *in, tensor_t *out, const float *scale, const float *shift);
void relu_f32(const tensor_t *in, tensor_t *out);

// fp32, fused: bn folded into w and b at export, relu clamps the output write
void conv2d_bn_relu_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p);
void depthwise_bn_relu_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b, const conv_params_t *p);

void avgpool_f32(const tensor_t *in, tensor_t *out);
void dense_f32(const tensor_t *in, tensor_t *out, const float *w, const float *b);

// int8: requant and relu live in each layer's params, so every op is fused
void conv2d_s8(const tensor_t *in, tensor_t *out, const layer_q_t *l, const conv_params_t *p);
void depthwise_s8(const tensor_t *in, tensor_t *out, const layer_q_t *l, const conv_params_t *p);
void avgpool_s8(const tensor_t *in, tensor_t *out);
void dense_s8(const tensor_t *in, tensor_t *out, const layer_q_t *l);

#endif
