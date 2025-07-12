#ifndef SGD_H
#define SGD_H

#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "memory/tensor/tensor_allocator.h"

struct sgd_optimizer
{
    size_t size;
    struct model_params *params;
    struct tensor *prev_b_t[MODEL_MAX_PARAMS];
    struct tensor_allocator *allocator;
};

cgrad_error sgd_optimizer_step(struct sgd_optimizer* opt, double lr, double momentum, bool nesterov);
cgrad_error sgd_optimizer_init(struct sgd_optimizer *opt, struct model_params *const params, struct tensor_allocator *allocator);
void sgd_optimizer_cleanup(struct sgd_optimizer *opt);

#endif