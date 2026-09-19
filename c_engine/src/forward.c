#include <string.h>

#include "forward.h"
#include "geometry.h"
#include "mfcc.h"

#if EI_WITH_F32
#include "weights_f32.h"
#endif
#if EI_WITH_UNFUSED
#include "weights_raw.h"
#endif
#if EI_WITH_INT8
#include "weights_int8.h"
#endif

static const char *const op_names[OP_COUNT] = {
    "conv2d", "depthwise_conv", "pointwise_conv", "batch_norm", "relu", "pool", "dense",
};

const char *ei_op_name(ei_op_t op)
{
    return op_names[op];
}

const char *ei_layer_name(int layer)
{
    return layer >= 0 && layer < EI_LAYERS ? ei_layer_names[layer] : "";
}

// chain graph: step i reads tensor i and writes tensor i + 1
static void add_step(ei_model_t *m, ei_op_t op, int layer, int h, int w, int c)
{
    int i = m->n_steps++;
    m->steps[i] = (step_t){op, layer, i, i + 1};
    m->tensors[i + 1] = (tensor_t){h, w, c, NULL, 1.0f, 0};
}

static int compiled_in(ei_mode_t mode)
{
    return (mode == EI_FUSED && EI_WITH_F32) || (mode == EI_UNFUSED && EI_WITH_UNFUSED) ||
           (mode == EI_INT8 && EI_WITH_INT8);
}

#if EI_WITH_INT8
// every int8 tensor carries the quant params of the layer that wrote it,
// pool keeps its input's
static void set_quant_params(ei_model_t *m)
{
    m->tensors[0].scale = EI_IN_SCALE;
    m->tensors[0].zero_point = EI_IN_ZP;
    for (int i = 0; i < m->n_steps; i++) {
        const step_t *s = &m->steps[i];
        tensor_t *out = &m->tensors[s->out];
        if (s->op == OP_POOL) {
            out->scale = m->tensors[s->in].scale;
            out->zero_point = m->tensors[s->in].zero_point;
            continue;
        }
        const layer_q_t *l = s->op == OP_DENSE ? &ei_int8_fc : &ei_int8_layers[s->layer];
        out->scale = l->out_scale;
        out->zero_point = l->out_zp;
    }
}
#endif

int32_t ei_init(ei_model_t *m, ei_mode_t mode, int naive, uint8_t *arena)
{
    if (!compiled_in(mode))
        return -1;
    memset(m, 0, sizeof *m);
    m->mode = mode;

    int h = EI_IN_H, w = EI_IN_W;
    m->tensors[0] = (tensor_t){h, w, 1, NULL, 1.0f, 0};
    for (int l = 0; l < EI_LAYERS; l++) {
        const conv_params_t *g = &ei_geom[l];
        h = (h + 2 * g->ph - g->kh) / g->sh + 1;
        w = (w + 2 * g->pw - g->kw) / g->sw + 1;
        ei_op_t op = l == 0 ? OP_CONV : g->kh == 1 && g->kw == 1 ? OP_PW : OP_DW;
        add_step(m, op, l, h, w, EI_WIDTH);
        if (mode == EI_UNFUSED) {
            add_step(m, OP_BN, l, h, w, EI_WIDTH);
            add_step(m, OP_RELU, l, h, w, EI_WIDTH);
        }
    }
    add_step(m, OP_POOL, -1, 1, 1, EI_WIDTH);
    add_step(m, OP_DENSE, -1, 1, 1, EI_CLASSES);

    // sizes rounded to 4 bytes keep every float tensor aligned in the arena
    int esize = mode == EI_INT8 ? 1 : 4;
    int step_in[EI_MAX_STEPS], step_out[EI_MAX_STEPS];
    m->plan.n_tensors = m->n_steps + 1;
    for (int i = 0; i < m->plan.n_tensors; i++)
        m->plan.tensors[i].size = (tensor_len(&m->tensors[i]) * esize + 3) & ~3;
    for (int i = 0; i < m->n_steps; i++) {
        step_in[i] = m->steps[i].in;
        step_out[i] = m->steps[i].out;
    }
    mem_liveness(&m->plan, step_in, step_out, m->n_steps);
    if (naive)
        mem_plan_naive(&m->plan);
    else if (mem_plan_greedy(&m->plan) < 0)
        return -1;

    if (arena)
        for (int i = 0; i < m->plan.n_tensors; i++)
            m->tensors[i].data = arena + m->plan.tensors[i].offset;
#if EI_WITH_INT8
    if (mode == EI_INT8)
        set_quant_params(m);
#endif
    return m->plan.arena_bytes;
}

#if EI_WITH_F32
static void run_fused(const step_t *s, const tensor_t *in, tensor_t *out)
{
    switch (s->op) {
    case OP_CONV:
    case OP_PW:
        conv2d_bn_relu_f32(in, out, ei_f32_layers[s->layer].w, ei_f32_layers[s->layer].b, &ei_geom[s->layer]);
        break;
    case OP_DW:
        depthwise_bn_relu_f32(in, out, ei_f32_layers[s->layer].w, ei_f32_layers[s->layer].b, &ei_geom[s->layer]);
        break;
    case OP_POOL:
        avgpool_f32(in, out);
        break;
    case OP_DENSE:
        dense_f32(in, out, fc_w, fc_b);
        break;
    default:
        break;
    }
}
#endif

#if EI_WITH_UNFUSED
static void run_unfused(const step_t *s, const tensor_t *in, tensor_t *out)
{
    const layer_raw_t *l = s->layer >= 0 ? &ei_raw_layers[s->layer] : NULL;
    switch (s->op) {
    case OP_CONV:
    case OP_PW:
        conv2d_f32(in, out, l->w, NULL, &ei_geom[s->layer]);
        break;
    case OP_DW:
        depthwise_f32(in, out, l->w, NULL, &ei_geom[s->layer]);
        break;
    case OP_BN:
        batchnorm_f32(in, out, l->bn_scale, l->bn_shift);
        break;
    case OP_RELU:
        relu_f32(in, out);
        break;
    case OP_POOL:
        avgpool_f32(in, out);
        break;
    case OP_DENSE:
        dense_f32(in, out, fc_w, fc_b);
        break;
    default:
        break;
    }
}
#endif

#if EI_WITH_INT8
static void run_int8(const step_t *s, const tensor_t *in, tensor_t *out)
{
    switch (s->op) {
    case OP_CONV:
    case OP_PW:
        conv2d_s8(in, out, &ei_int8_layers[s->layer], &ei_geom[s->layer]);
        break;
    case OP_DW:
        depthwise_s8(in, out, &ei_int8_layers[s->layer], &ei_geom[s->layer]);
        break;
    case OP_POOL:
        avgpool_s8(in, out);
        break;
    case OP_DENSE:
        dense_s8(in, out, &ei_int8_fc);
        break;
    default:
        break;
    }
}
#endif

void ei_run(ei_model_t *m)
{
    for (int i = 0; i < m->n_steps; i++) {
        const step_t *s = &m->steps[i];
        const tensor_t *in = &m->tensors[s->in];
        tensor_t *out = &m->tensors[s->out];
        uint64_t start = ei_clock();
#if EI_WITH_F32
        if (m->mode == EI_FUSED)
            run_fused(s, in, out);
#endif
#if EI_WITH_UNFUSED
        if (m->mode == EI_UNFUSED)
            run_unfused(s, in, out);
#endif
#if EI_WITH_INT8
        if (m->mode == EI_INT8)
            run_int8(s, in, out);
#endif
        m->ticks[i] += ei_clock() - start;
        if (m->hook)
            m->hook(m, i, m->hook_ctx);
    }
}

void ei_set_features(ei_model_t *m, const float *x)
{
    tensor_t *in = &m->tensors[0];
    if (m->mode == EI_INT8)
        quantize_tensor(x, in);
    else
        memcpy(in->data, x, tensor_len(in) * sizeof(float));
}

void ei_set_mfcc(ei_model_t *m, const int32_t *mfcc_q16)
{
    tensor_t *in = &m->tensors[0];
#if EI_WITH_INT8
    if (m->mode == EI_INT8) {
        mfcc_to_s8(mfcc_q16, in->data, ei_mfcc_mean_q16, ei_mfcc_mult, ei_mfcc_shift, EI_IN_ZP);
        return;
    }
#endif
#if EI_WITH_F32
    float *x = in->data;
    for (int i = 0; i < tensor_len(in); i++) {
        int k = i % EI_IN_W;
        x[i] = ((float)mfcc_q16[i] / 65536.0f - ei_feat_mean[k]) / ei_feat_std[k];
    }
#endif
}

// weight bytes a mode ships: what lands in flash, not the host binary size
int32_t ei_weight_bytes(ei_mode_t mode)
{
    int32_t total = 0;
    for (int l = 0; l < EI_LAYERS; l++) {
        const conv_params_t *g = &ei_geom[l];
        // depthwise keeps one tap per channel, pointwise mixes all of them
        int cin = l == 0 ? 1 : (g->kh == 1 && g->kw == 1) ? EI_WIDTH : 1;
        int weights = g->kh * g->kw * cin * EI_WIDTH;
        if (mode == EI_INT8)
            total += weights + EI_WIDTH * (int)(sizeof(int32_t) * 2 + 1);
        else
            total += (weights + EI_WIDTH * (mode == EI_UNFUSED ? 2 : 1)) * (int)sizeof(float);
    }
    int fc = EI_CLASSES * EI_WIDTH;
    total += mode == EI_INT8 ? fc + EI_CLASSES * (int)(sizeof(int32_t) * 2 + 1)
                             : (fc + EI_CLASSES) * (int)sizeof(float);
    return total;
}

void ei_get_logits(const ei_model_t *m, float *out)
{
    const tensor_t *y = &m->tensors[m->n_steps];
    if (m->mode == EI_INT8)
        dequantize_tensor(y, out);
    else
        memcpy(out, y->data, EI_CLASSES * sizeof(float));
}
