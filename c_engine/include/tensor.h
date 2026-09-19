#ifndef EI_TENSOR_H
#define EI_TENSOR_H

#include <stdint.h>

// batch 1, hwc layout: channel is the fastest-moving index, so a pixel's
// channels are contiguous and 1x1 convs are plain row-major matmuls
typedef struct {
    int h, w, c;
    void *data;
    // int8 tensors only
    float scale;
    int32_t zero_point;
} tensor_t;

static inline int tensor_len(const tensor_t *t)
{
    return t->h * t->w * t->c;
}

#endif
