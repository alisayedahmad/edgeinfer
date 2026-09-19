#include "memory.h"

void mem_liveness(mem_plan_t *plan, const int *step_in, const int *step_out, int n_steps)
{
    for (int t = 0; t < plan->n_tensors; t++) {
        plan->tensors[t].first = -1;
        plan->tensors[t].last = -1;
    }
    for (int s = 0; s < n_steps; s++) {
        plan->tensors[step_out[s]].first = s;
        if (plan->tensors[step_in[s]].last < s)
            plan->tensors[step_in[s]].last = s;
    }
    plan->tensors[plan->n_tensors - 1].last = n_steps;
}

void mem_plan_naive(mem_plan_t *plan)
{
    int32_t offset = 0;
    for (int t = 0; t < plan->n_tensors; t++) {
        plan->buffers[t].size = plan->tensors[t].size;
        plan->buffers[t].offset = offset;
        plan->tensors[t].buffer = t;
        plan->tensors[t].offset = offset;
        offset += plan->tensors[t].size;
    }
    plan->n_buffers = plan->n_tensors;
    plan->arena_bytes = offset;
}

int mem_plan_greedy(mem_plan_t *plan)
{
    int n = plan->n_tensors;
    int order[MEM_MAX_TENSORS], owner[MEM_MAX_BUFFERS];

    // birth order, stable insertion sort on first write
    for (int i = 0; i < n; i++) {
        int j = i;
        for (; j > 0 && plan->tensors[order[j - 1]].first > plan->tensors[i].first; j--)
            order[j] = order[j - 1];
        order[j] = i;
    }

    plan->n_buffers = 0;
    for (int i = 0; i < n; i++) {
        mem_tensor_t *t = &plan->tensors[order[i]];
        int fit = -1, grow = -1;
        for (int b = 0; b < plan->n_buffers; b++) {
            // a tensor read by the step that writes t is still live during it
            if (plan->tensors[owner[b]].last >= t->first)
                continue;
            int32_t size = plan->buffers[b].size;
            if (size >= t->size && (fit < 0 || size < plan->buffers[fit].size))
                fit = b;
            if (grow < 0 || size > plan->buffers[grow].size)
                grow = b;
        }
        int b = fit >= 0 ? fit : grow;
        if (b < 0) {
            if (plan->n_buffers == MEM_MAX_BUFFERS)
                return -1;
            b = plan->n_buffers++;
            plan->buffers[b].size = 0;
        }
        if (plan->buffers[b].size < t->size)
            plan->buffers[b].size = t->size;
        owner[b] = order[i];
        t->buffer = b;
    }

    int32_t offset = 0;
    for (int b = 0; b < plan->n_buffers; b++) {
        plan->buffers[b].offset = offset;
        offset += plan->buffers[b].size;
    }
    for (int i = 0; i < n; i++)
        plan->tensors[i].offset = plan->buffers[plan->tensors[i].buffer].offset;
    plan->arena_bytes = offset;
    return 0;
}
