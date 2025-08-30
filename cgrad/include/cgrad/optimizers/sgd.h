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
    double lr;
    double momemtum;
    bool nesterov;
};

cgrad_error sgd_optimizer_init(struct sgd_optimizer *opt, struct model_params *const params, const double lr, const double momentum, const bool nesterov, struct cgrad_env *env);
void sgd_optimizer_cleanup(struct sgd_optimizer *opt);
cgrad_error sgd_optimizer_step(struct sgd_optimizer *opt);
static inline void sgd_optimizer_zero_grad(struct sgd_optimizer *opt);

static inline void sgd_optimizer_zero_grad(struct sgd_optimizer *opt)
{
    if (!opt)
    {
        return;
    }

    model_params_zero_grad(opt->params);
}

#endif