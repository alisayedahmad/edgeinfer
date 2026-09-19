// host cli around the engine. python drives it with raw float32 / int16
// files and reads back binaries or json, see runtime/c_runner.py
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "forward.h"
#include "mfcc.h"

#define FEATURES (EI_IN_H * EI_IN_W)

uint64_t ei_clock(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void die(const char *msg)
{
    fprintf(stderr, "edgeinfer: %s\n", msg);
    exit(1);
}

static void *read_file(const char *path, size_t bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot open input file");
    void *buf = malloc(bytes);
    if (!buf || fread(buf, 1, bytes, f) != bytes)
        die("input file shorter than expected");
    fclose(f);
    return buf;
}

static ei_mode_t parse_mode(const char *s)
{
    if (!strcmp(s, "fp32"))
        return EI_FUSED;
    if (!strcmp(s, "fp32-unfused"))
        return EI_UNFUSED;
    if (!strcmp(s, "int8"))
        return EI_INT8;
    die("mode is fp32, fp32-unfused or int8");
    return EI_FUSED;
}

static uint8_t *setup(ei_model_t *m, ei_mode_t mode, int naive)
{
    int32_t bytes = ei_init(m, mode, naive, NULL);
    if (bytes < 0)
        die("mode not compiled in");
    uint8_t *arena = calloc(1, bytes);
    ei_init(m, mode, naive, arena);
    return arena;
}

// tensor 0 is the input, tensor i + 1 is written by step i
static void tensor_name(const ei_model_t *m, int t, char *buf, size_t size)
{
    if (t == 0) {
        snprintf(buf, size, "input");
        return;
    }
    const step_t *s = &m->steps[t - 1];
    const char *suffix = s->op == OP_BN ? "_bn" : s->op == OP_RELU ? "_relu" : "";
    if (s->op == OP_POOL || s->op == OP_DENSE)
        snprintf(buf, size, "%s", s->op == OP_POOL ? "pool" : "fc");
    else
        snprintf(buf, size, "%s%s", ei_layer_name(s->layer), suffix);
}

static void cmd_eval(ei_model_t *m, const char *in_path, int n, const char *out_path)
{
    float *x = read_file(in_path, (size_t)n * FEATURES * sizeof(float));
    FILE *out = fopen(out_path, "wb");
    if (!out)
        die("cannot open output file");
    float logits[EI_CLASSES];
    for (int i = 0; i < n; i++) {
        ei_set_features(m, x + (size_t)i * FEATURES);
        ei_run(m);
        ei_get_logits(m, logits);
        fwrite(logits, sizeof logits, 1, out);
    }
    fclose(out);
    free(x);
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// per-step and end-to-end timing as json, input values don't change the cost
static void cmd_bench(ei_model_t *m, const char *mode, int runs)
{
    float x[FEATURES];
    for (int i = 0; i < FEATURES; i++)
        x[i] = (float)((i * 37) % 17) / 8.0f - 1.0f;
    ei_set_features(m, x);
    for (int i = 0; i < 20; i++)
        ei_run(m);
    memset(m->ticks, 0, sizeof m->ticks);

    uint64_t *lat = malloc(runs * sizeof *lat);
    for (int i = 0; i < runs; i++) {
        uint64_t start = ei_clock();
        ei_run(m);
        lat[i] = ei_clock() - start;
    }
    qsort(lat, runs, sizeof *lat, cmp_u64);
    double mean = 0;
    for (int i = 0; i < runs; i++)
        mean += (double)lat[i] / runs;

    printf("{\"mode\": \"%s\", \"runs\": %d, \"arena_bytes\": %d,\n", mode, runs, (int)m->plan.arena_bytes);
    printf(" \"latency_ns\": {\"p50\": %llu, \"p90\": %llu, \"p99\": %llu, \"mean\": %.1f},\n",
           (unsigned long long)lat[runs / 2], (unsigned long long)lat[runs * 9 / 10],
           (unsigned long long)lat[runs * 99 / 100], mean);
    printf(" \"steps\": [\n");
    char name[32];
    for (int i = 0; i < m->n_steps; i++) {
        tensor_name(m, i + 1, name, sizeof name);
        printf("  {\"name\": \"%s\", \"op\": \"%s\", \"ns\": %.1f, \"out_bytes\": %d}%s\n", name,
               ei_op_name(m->steps[i].op), (double)m->ticks[i] / runs, (int)m->plan.tensors[i + 1].size,
               i + 1 < m->n_steps ? "," : "");
    }
    printf(" ]}\n");
    free(lat);
}

static void cmd_plan(const ei_model_t *m, const char *mode, int naive)
{
    printf("{\"mode\": \"%s\", \"planner\": \"%s\", \"arena_bytes\": %d, \"weight_bytes\": %d, "
           "\"n_buffers\": %d, \"n_steps\": %d,\n",
           mode, naive ? "naive" : "greedy", (int)m->plan.arena_bytes, (int)ei_weight_bytes(m->mode),
           m->plan.n_buffers, m->n_steps);
    printf(" \"tensors\": [\n");
    char name[32];
    for (int t = 0; t < m->plan.n_tensors; t++) {
        const mem_tensor_t *x = &m->plan.tensors[t];
        tensor_name(m, t, name, sizeof name);
        printf("  {\"name\": \"%s\", \"size\": %d, \"first\": %d, \"last\": %d, \"buffer\": %d, \"offset\": %d}%s\n",
               name, (int)x->size, x->first, x->last, x->buffer, (int)x->offset,
               t + 1 < m->plan.n_tensors ? "," : "");
    }
    printf(" ]}\n");
}

typedef struct {
    FILE *out;
    float *buf;
} tap_ctx_t;

// every step's output as float, dequantized in int8 mode
static void dump_step(const ei_model_t *m, int step, void *ctx)
{
    tap_ctx_t *tap = ctx;
    const tensor_t *t = &m->tensors[m->steps[step].out];
    int n = tensor_len(t);
    if (m->mode == EI_INT8)
        dequantize_tensor(t, tap->buf);
    else
        memcpy(tap->buf, t->data, n * sizeof(float));
    fwrite(tap->buf, sizeof(float), n, tap->out);
}

static void cmd_taps(ei_model_t *m, const char *in_path, int n, const char *out_path)
{
    float *x = read_file(in_path, (size_t)n * FEATURES * sizeof(float));
    int largest = 0;
    for (int t = 0; t < m->plan.n_tensors; t++)
        if (tensor_len(&m->tensors[t]) > largest)
            largest = tensor_len(&m->tensors[t]);
    tap_ctx_t tap = {fopen(out_path, "wb"), malloc(largest * sizeof(float))};
    if (!tap.out)
        die("cannot open output file");
    m->hook = dump_step;
    m->hook_ctx = &tap;
    for (int i = 0; i < n; i++) {
        ei_set_features(m, x + (size_t)i * FEATURES);
        ei_run(m);
    }
    fclose(tap.out);
    free(tap.buf);
    free(x);
}

static void cmd_mfcc(const char *in_path, int n, const char *out_path)
{
    int16_t *pcm = read_file(in_path, (size_t)n * MFCC_SAMPLES * sizeof(int16_t));
    FILE *out = fopen(out_path, "wb");
    if (!out)
        die("cannot open output file");
    int32_t q16[FEATURES];
    float db[FEATURES];
    for (int i = 0; i < n; i++) {
        mfcc_q16(pcm + (size_t)i * MFCC_SAMPLES, q16);
        for (int k = 0; k < FEATURES; k++)
            db[k] = (float)q16[k] / 65536.0f;
        fwrite(db, sizeof db, 1, out);
    }
    fclose(out);
    free(pcm);
}

// arena sizes the cortex-m4 build declares statically
static void cmd_header(void)
{
    ei_model_t m;
    printf("// generated by edgeinfer header, do not edit\n");
    printf("#define EI_ARENA_INT8 %d\n", (int)ei_init(&m, EI_INT8, 0, NULL));
    printf("#define EI_ARENA_FP32 %d\n", (int)ei_init(&m, EI_FUSED, 0, NULL));
}

static const char usage[] =
    "usage: edgeinfer eval  <mode> <features.f32> <n> <logits.f32>\n"
    "       edgeinfer bench <mode> [runs]\n"
    "       edgeinfer plan  <mode> [naive]\n"
    "       edgeinfer taps  <mode> <features.f32> <n> <taps.f32>\n"
    "       edgeinfer mfcc  <pcm.s16> <n> <mfcc.f32>\n"
    "       edgeinfer header\n"
    "mode is fp32, fp32-unfused or int8\n";

int main(int argc, char **argv)
{
    static ei_model_t m;
    if (argc >= 2 && !strcmp(argv[1], "header")) {
        cmd_header();
        return 0;
    }
    if (argc == 5 && !strcmp(argv[1], "mfcc")) {
        cmd_mfcc(argv[2], atoi(argv[3]), argv[4]);
        return 0;
    }
    if (argc < 3) {
        fputs(usage, stderr);
        return 1;
    }
    const char *cmd = argv[1], *mode = argv[2];
    int naive = !strcmp(cmd, "plan") && argc > 3 && !strcmp(argv[3], "naive");
    uint8_t *arena = setup(&m, parse_mode(mode), naive);
    if (!strcmp(cmd, "eval") && argc == 6)
        cmd_eval(&m, argv[3], atoi(argv[4]), argv[5]);
    else if (!strcmp(cmd, "bench"))
        cmd_bench(&m, mode, argc > 3 ? atoi(argv[3]) : 1000);
    else if (!strcmp(cmd, "plan"))
        cmd_plan(&m, mode, naive);
    else if (!strcmp(cmd, "taps") && argc == 6)
        cmd_taps(&m, argv[3], atoi(argv[4]), argv[5]);
    else {
        fputs(usage, stderr);
        return 1;
    }
    free(arena);
    return 0;
}
