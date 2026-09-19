#ifndef EI_FORWARD_H
#define EI_FORWARD_H

#include <stdint.h>

#include "memory.h"
#include "model.h"
#include "ops.h"
#include "quantize.h"
#include "tensor.h"

// which weight sets get compiled in. the cortex-m4 build drops the unfused
// copy to stay inside 1 mb of flash
#ifndef EI_WITH_F32
#define EI_WITH_F32 1
#endif
#ifndef EI_WITH_UNFUSED
#define EI_WITH_UNFUSED 1
#endif
#ifndef EI_WITH_INT8
#define EI_WITH_INT8 1
#endif
#if EI_WITH_UNFUSED && !EI_WITH_F32
#error "the unfused path needs the fp32 fc weights, set EI_WITH_F32"
#endif

typedef enum { EI_FUSED, EI_UNFUSED, EI_INT8 } ei_mode_t;

// op kinds double as profiling categories
typedef enum { OP_CONV, OP_DW, OP_PW, OP_BN, OP_RELU, OP_POOL, OP_DENSE, OP_COUNT } ei_op_t;

typedef struct {
    const float *w, *b;
} layer_f32_t;

typedef struct {
    const float *w, *bn_scale, *bn_shift;
} layer_raw_t;

typedef struct {
    ei_op_t op;
    // index into the layer tables, -1 for pool and dense
    int layer;
    int in, out;
} step_t;

// unfused worst case: conv, bn and relu per layer, then pool and dense
#define EI_MAX_STEPS (3 * EI_LAYERS + 2)

typedef struct ei_model ei_model_t;
typedef void (*ei_hook_t)(const ei_model_t *m, int step, void *ctx);

struct ei_model {
    ei_mode_t mode;
    int n_steps;
    step_t steps[EI_MAX_STEPS];
    tensor_t tensors[EI_MAX_STEPS + 1];
    mem_plan_t plan;
    // per-step clock ticks, accumulated across runs
    uint64_t ticks[EI_MAX_STEPS];
    // called after every step, e.g. to dump intermediate tensors
    ei_hook_t hook;
    void *hook_ctx;
};

// build the step list and memory plan for a mode. returns the arena bytes
// the plan needs, or -1 if the mode's weights are not compiled in.
// call with arena = NULL to size it, then again with the real arena
int32_t ei_init(ei_model_t *m, ei_mode_t mode, int naive, uint8_t *arena);
void ei_run(ei_model_t *m);

// normalized float features in, quantized on the way in int8 mode
void ei_set_features(ei_model_t *m, const float *x);
// q16 mfcc from mfcc_q16() in: normalized, and quantized in int8 mode
void ei_set_mfcc(ei_model_t *m, const int32_t *mfcc_q16);
// float logits out, dequantized in int8 mode
void ei_get_logits(const ei_model_t *m, float *out);

// weight bytes a mode ships, the flash figure for the target
int32_t ei_weight_bytes(ei_mode_t mode);

const char *ei_layer_name(int layer);
const char *ei_op_name(ei_op_t op);

// provided by the platform: host counts ns, the cortex-m4 build systick ticks
uint64_t ei_clock(void);

#endif
