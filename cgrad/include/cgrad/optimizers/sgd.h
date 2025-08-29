#ifndef SGD_H
#define SGD_H

#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/model/model_params.h"
#include "cgrad/cgrad_env.h"

struct sgd_optimizer
{
    size_t size;
    struct model_params *params;
    struct tensor *prev_b_t[MODEL_MAX_PARAMS];
    struct tensor_allocator *tensor_alloc;
};

cgrad_error sgd_optimizer_step(struct sgd_optimizer* opt, double lr, double momentum, bool nesterov);
cgrad_error sgd_optimizer_init(struct sgd_optimizer *opt, struct model_params *const params, struct cgrad_env *env);
void sgd_optimizer_cleanup(struct sgd_optimizer *opt);

#endif