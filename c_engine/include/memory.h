#ifndef EI_MEMORY_H
#define EI_MEMORY_H

#include <stdint.h>

// the naive plan gives every tensor its own buffer
#define MEM_MAX_TENSORS 64
#define MEM_MAX_BUFFERS MEM_MAX_TENSORS

typedef struct {
    int32_t size;
    // step that writes it (-1 for the graph input), last step that reads it
    int first, last;
    // physical buffer backing it and its byte offset in the arena
    int buffer;
    int32_t offset;
} mem_tensor_t;

typedef struct {
    int32_t size, offset;
} mem_buffer_t;

typedef struct {
    int n_tensors, n_buffers;
    mem_tensor_t tensors[MEM_MAX_TENSORS];
    mem_buffer_t buffers[MEM_MAX_BUFFERS];
    int32_t arena_bytes;
} mem_plan_t;

// first write and last read per tensor, from each step's input and output
// tensor ids. the last tensor is the graph output and stays live to the end
void mem_liveness(mem_plan_t *plan, const int *step_in, const int *step_out, int n_steps);

// one buffer per tensor, the no-reuse baseline
void mem_plan_naive(mem_plan_t *plan);

// walk tensors in birth order, reuse a buffer whose tensor is already dead:
// smallest one that fits, else grow the largest free one, else add a buffer.
// returns -1 if it runs out of buffers
int mem_plan_greedy(mem_plan_t *plan);

#endif
