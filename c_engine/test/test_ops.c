// operator-level tests. fp32 ops are checked against pytorch within a float
// tolerance, int8 ops must match train/quant.py bit for bit. vectors.h comes
// from gen_vectors.py
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "memory.h"
#include "mfcc.h"
#include "ops.h"
#include "quantize.h"
#include "vectors.h"

#define ATOL 1e-4f
// fixed-point mfcc vs the float reference, in db. the error lives in quiet
// mel bands where 16-bit fft rounding noise is comparable to the signal
#define MFCC_MAX_ERR 0.5f
#define MFCC_MEAN_ERR 0.05f
#define LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

static int failures;

static void check(int ok, const char *name, const char *detail)
{
    printf("%-4s %-28s %s\n", ok ? "ok" : "FAIL", name, detail);
    failures += !ok;
}

static float max_diff(const float *a, const float *b, int n)
{
    float worst = 0.0f;
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > worst)
            worst = fabsf(a[i] - b[i]);
    return worst;
}

static void check_f32(const char *name, const float *got, const float *want, int n)
{
    char detail[64];
    float err = max_diff(got, want, n);
    snprintf(detail, sizeof detail, "max |diff| %.2e", err);
    check(err <= ATOL, name, detail);
}

static void check_s8(const char *name, const int8_t *got, const int8_t *want, int n)
{
    int bad = 0;
    for (int i = 0; i < n; i++)
        bad += got[i] != want[i];
    char detail[64];
    snprintf(detail, sizeof detail, "%d/%d mismatched", bad, n);
    check(bad == 0, name, detail);
}

static tensor_t tensor(int h, int w, int c, void *data)
{
    return (tensor_t){h, w, c, data, 1.0f, 0};
}

static void test_fp32(void)
{
    static float a[25 * 5 * 16], b[25 * 5 * 16];
    const conv_params_t conv1 = {10, 4, 2, 2, 5, 1}, dw = {3, 3, 1, 1, 1, 1}, pw = {1, 1, 1, 1, 0, 0};

    tensor_t x = tensor(49, 10, 1, (void *)conv_x), y = tensor(25, 5, 8, a);
    conv2d_f32(&x, &y, conv_w, conv_b, &conv1);
    check_f32("conv2d 10x4 stride 2", a, conv_y, LEN(conv_y));

    x = tensor(25, 5, 12, (void *)act_x);
    y = tensor(25, 5, 16, a);
    conv2d_f32(&x, &y, pw_w, pw_b, &pw);
    check_f32("conv2d 1x1", a, pw_y, LEN(pw_y));

    y = tensor(25, 5, 12, a);
    depthwise_f32(&x, &y, dw_w, dw_b, &dw);
    check_f32("depthwise 3x3", a, dw_y, LEN(dw_y));
    batchnorm_f32(&x, &y, bn_scale, bn_shift);
    check_f32("batchnorm", a, bn_y, LEN(bn_y));
    relu_f32(&x, &y);
    check_f32("relu", a, relu_y, LEN(relu_y));

    tensor_t p = tensor(1, 1, 12, a), fc = tensor(1, 1, 35, b);
    avgpool_f32(&x, &p);
    check_f32("avgpool", a, pool_y, LEN(pool_y));
    dense_f32(&p, &fc, fc_w, fc_b);
    check_f32("dense", b, fc_y, LEN(fc_y));
}

// fused kernel vs pytorch, and vs the unfused c path: bn folding is exact
// algebra, what's left is float rounding from the different evaluation order
static void test_fusion(void)
{
    static float fused[25 * 5 * 12], t1[25 * 5 * 12], t2[25 * 5 * 12];
    const conv_params_t dw = {3, 3, 1, 1, 1, 1}, pw = {1, 1, 1, 1, 0, 0};
    tensor_t x = tensor(25, 5, 12, (void *)act_x);
    tensor_t yf = tensor(25, 5, 12, fused), y1 = tensor(25, 5, 12, t1), y2 = tensor(25, 5, 12, t2);
    char detail[64];

    depthwise_bn_relu_f32(&x, &yf, fused_dw_w, bn_shift, &dw);
    check_f32("fused depthwise+bn+relu", fused, fused_dw_y, LEN(fused_dw_y));
    depthwise_f32(&x, &y1, dw_w, NULL, &dw);
    batchnorm_f32(&y1, &y2, bn_scale, bn_shift);
    relu_f32(&y2, &y1);
    snprintf(detail, sizeof detail, "max |fused - unfused| %.2e", max_diff(fused, t1, LEN(t1)));
    check(max_diff(fused, t1, LEN(t1)) <= ATOL, "fusion is exact (depthwise)", detail);

    conv2d_bn_relu_f32(&x, &yf, fused_pw_w, bn_shift, &pw);
    check_f32("fused conv1x1+bn+relu", fused, fused_pw_y, LEN(fused_pw_y));
    conv2d_f32(&x, &y1, fused_pw_raw_w, NULL, &pw);
    batchnorm_f32(&y1, &y2, bn_scale, bn_shift);
    relu_f32(&y2, &y1);
    snprintf(detail, sizeof detail, "max |fused - unfused| %.2e", max_diff(fused, t1, LEN(t1)));
    check(max_diff(fused, t1, LEN(t1)) <= ATOL, "fusion is exact (1x1)", detail);
}

static void test_int8(void)
{
    static int8_t out[25 * 5 * 16], pooled[12], logits[35];
    const conv_params_t conv1 = {10, 4, 2, 2, 5, 1}, dw = {3, 3, 1, 1, 1, 1}, pw = {1, 1, 1, 1, 0, 0};

    tensor_t x = tensor(49, 10, 1, (void *)q_conv_x), y = tensor(25, 5, 8, out);
    conv2d_s8(&x, &y, &q_conv_layer, &conv1);
    check_s8("conv2d_s8 10x4 stride 2", out, q_conv_y, LEN(q_conv_y));

    x = tensor(25, 5, 12, (void *)q_x);
    y = tensor(25, 5, 12, out);
    depthwise_s8(&x, &y, &q_dw_layer, &dw);
    check_s8("depthwise_s8", out, q_dw_y, LEN(q_dw_y));
    y = tensor(25, 5, 16, out);
    conv2d_s8(&x, &y, &q_pw_layer, &pw);
    check_s8("conv2d_s8 1x1 (qmatmul)", out, q_pw_y, LEN(q_pw_y));

    tensor_t p = tensor(1, 1, 12, pooled), fc = tensor(1, 1, 35, logits);
    avgpool_s8(&x, &p);
    check_s8("avgpool_s8", pooled, q_pool_y, LEN(q_pool_y));
    dense_s8(&p, &fc, &q_fc_layer);
    check_s8("dense_s8", logits, q_fc_y, LEN(q_fc_y));
}

static void test_quant(void)
{
    static int8_t q[490];
    static float back[490];
    tensor_t t = {49, 10, 1, q, 0.0625f, 5};
    quantize_tensor(quant_x, &t);
    check_s8("quantize_tensor", q, quant_y, LEN(quant_y));

    // round trip error is at most half a step inside the representable range
    dequantize_tensor(&t, back);
    float worst = 0.0f;
    for (int i = 0; i < LEN(quant_x); i++) {
        float lo = (-128 - 5) * 0.0625f, hi = (127 - 5) * 0.0625f;
        if (quant_x[i] > lo && quant_x[i] < hi && fabsf(back[i] - quant_x[i]) > worst)
            worst = fabsf(back[i] - quant_x[i]);
    }
    char detail[64];
    snprintf(detail, sizeof detail, "max err %.4f, step 0.0625", worst);
    check(worst <= 0.5f * 0.0625f + 1e-6f, "dequantize round trip", detail);

    int bad = 0;
    for (int i = 0; i < LEN(rq_acc); i++)
        bad += requant(rq_acc[i], rq_mult[i], rq_shift[i]) != rq_y[i];
    snprintf(detail, sizeof detail, "%d/%d mismatched", bad, LEN(rq_acc));
    check(bad == 0, "requant", detail);
}

static void test_memory(void)
{
    // chain input -> t1 -> ... -> t5, sizes chosen so best fit and grow both happen
    const int32_t sizes[] = {100, 400, 400, 200, 400, 50};
    const int step_in[] = {0, 1, 2, 3, 4}, step_out[] = {1, 2, 3, 4, 5};
    static mem_plan_t plan;
    plan.n_tensors = LEN(sizes);
    for (int i = 0; i < LEN(sizes); i++)
        plan.tensors[i].size = sizes[i];
    mem_liveness(&plan, step_in, step_out, LEN(step_in));

    mem_plan_naive(&plan);
    check(plan.arena_bytes == 1550, "naive plan is the sum", "");

    mem_plan_greedy(&plan);
    char detail[64];
    snprintf(detail, sizeof detail, "arena %d bytes, %d buffers", (int)plan.arena_bytes, plan.n_buffers);
    check(plan.arena_bytes == 800 && plan.n_buffers == 2, "greedy ping-pongs a chain", detail);

    // no two tensors sharing a buffer may be live at the same time
    int overlap = 0;
    for (int i = 0; i < plan.n_tensors; i++)
        for (int j = i + 1; j < plan.n_tensors; j++) {
            const mem_tensor_t *a = &plan.tensors[i], *b = &plan.tensors[j];
            overlap += a->buffer == b->buffer && a->first <= b->last && b->first <= a->last;
        }
    check(overlap == 0, "live tensors never share", "");
}

static void test_mfcc(void)
{
    static int32_t q16[490];
    static int8_t s8[490];
    mfcc_q16(mfcc_pcm, q16);
    float worst = 0.0f, mean = 0.0f;
    for (int i = 0; i < 490; i++) {
        float err = fabsf((float)q16[i] / 65536.0f - mfcc_ref[i]);
        worst = err > worst ? err : worst;
        mean += err / 490.0f;
    }
    char detail[64];
    snprintf(detail, sizeof detail, "max %.3f db, mean %.4f db", worst, mean);
    check(worst <= MFCC_MAX_ERR && mean <= MFCC_MEAN_ERR, "mfcc_q16 vs float reference", detail);

    mfcc_to_s8(m8_in, s8, m8_mean, m8_mult, m8_shift, M8_ZP);
    check_s8("mfcc_to_s8", s8, m8_y, LEN(m8_y));
}

int main(void)
{
    test_fp32();
    test_fusion();
    test_int8();
    test_quant();
    test_memory();
    test_mfcc();
    printf("%s: %d failure%s\n", failures ? "FAILED" : "passed", failures, failures == 1 ? "" : "s");
    return failures != 0;
}
