// cortex-m4 app for qemu: fixed-point mfcc on a test clip, then inference,
// timed per operator with systick. output is one json line per mode over
// semihosting. no printf, no heap, no fpu
#include <stdint.h>

#include "clip.h"
#include "forward.h"
#include "mfcc.h"
#include "plan.h"

#define SYST_CSR (*(volatile uint32_t *)0xe000e010)
#define SYST_RVR (*(volatile uint32_t *)0xe000e014)
#define SYST_CVR (*(volatile uint32_t *)0xe000e018)
#define SYST_MAX 0xffffff

#define ARENA_BYTES (EI_ARENA_FP32 > EI_ARENA_INT8 ? EI_ARENA_FP32 : EI_ARENA_INT8)

static uint8_t arena[ARENA_BYTES] __attribute__((aligned(8)));
static int32_t mfcc[EI_IN_H * EI_IN_W];
static ei_model_t model;
static volatile uint32_t overflows;

void SysTick_Handler(void)
{
    overflows++;
}

// systick counts down at the cpu clock, so ticks map to instructions when
// qemu runs with -icount
uint64_t ei_clock(void)
{
    uint32_t before, after, value;
    do {
        before = overflows;
        value = SYST_CVR;
        after = overflows;
    } while (before != after);
    return ((uint64_t)before << 24) + (SYST_MAX - value);
}

static int semihost(int op, void *arg)
{
    register int r0 __asm__("r0") = op;
    register void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

static void put(const char *s)
{
    semihost(0x04, (void *)s);
}

static void put_i64(int64_t v)
{
    char buf[24];
    int i = sizeof buf - 1;
    int neg = v < 0;
    uint64_t u = neg ? -(uint64_t)v : (uint64_t)v;
    buf[i] = 0;
    do {
        buf[--i] = '0' + (char)(u % 10);
        u /= 10;
    } while (u);
    if (neg)
        buf[--i] = '-';
    put(buf + i);
}

static void run(ei_mode_t mode, const char *name)
{
    if (ei_init(&model, mode, 0, arena) < 0)
        return;
    uint64_t start = ei_clock();
    mfcc_q16(clip_pcm, mfcc);
    uint64_t mfcc_ticks = ei_clock() - start;

    ei_set_mfcc(&model, mfcc);
    start = ei_clock();
    ei_run(&model);
    uint64_t infer_ticks = ei_clock() - start;

    static float logits[EI_CLASSES];
    ei_get_logits(&model, logits);
    int best = 0;
    for (int i = 1; i < EI_CLASSES; i++)
        if (logits[i] > logits[best])
            best = i;

    put("{\"mode\": \"");
    put(name);
    put("\", \"word\": \"");
    put(clip_word);
    put("\", \"label\": ");
    put_i64(CLIP_LABEL);
    put(", \"pred\": ");
    put_i64(best);
    put(", \"arena_bytes\": ");
    put_i64(model.plan.arena_bytes);
    put(", \"mfcc_ticks\": ");
    put_i64((int64_t)mfcc_ticks);
    put(", \"infer_ticks\": ");
    put_i64((int64_t)infer_ticks);
    put(", \"steps\": [");
    for (int i = 0; i < model.n_steps; i++) {
        ei_op_t op = model.steps[i].op;
        put(i ? ", [\"" : "[\"");
        // pool and dense have no layer index, name them after the op
        put(op == OP_POOL ? "pool" : op == OP_DENSE ? "fc" : ei_layer_name(model.steps[i].layer));
        put("\", \"");
        put(ei_op_name(op));
        put("\", ");
        put_i64((int64_t)model.ticks[i]);
        put("]");
    }
    put("]}\n");
}

int main(void)
{
    SYST_RVR = SYST_MAX;
    SYST_CVR = 0;
    // enable | tickint | clksource = processor
    SYST_CSR = 7;
    // writing cvr zeroes it, the first reload lands a tick later. read it as
    // zero before then and the clock starts a whole period ahead
    while (SYST_CVR == 0) {
    }

    run(EI_INT8, "int8");
    run(EI_FUSED, "fp32");

    // semihosting SYS_EXIT, application exit
    semihost(0x18, (void *)0x20026);
    return 0;
}
