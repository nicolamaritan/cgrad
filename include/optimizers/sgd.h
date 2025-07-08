#ifndef SGD_H
#define SGD_H

#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "memory/tensor_allocator.h"

struct sgd_optimizer
{
    size_t size;
    struct tensor *prev_b_t[MODEL_MAX_PARAMS];
    struct tensor_allocator *allocator;
};

void sgd_step(double lr, double momentum, bool nesterov, struct sgd_optimizer* state, struct model_params* params);
cgrad_error init_sgd_state(struct sgd_optimizer *state, const struct model_params *const params, struct tensor_allocator *allocator);
void free_sgd_state_tensors(struct sgd_optimizer *state);

#endif