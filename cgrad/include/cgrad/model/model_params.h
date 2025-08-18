#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/config.h"
#include <string.h>

struct model_params
{
    struct tensor *params[MODEL_MAX_PARAMS];
    size_t size;
};

void model_params_init(struct model_params *const params);
cgrad_error add_model_param(struct model_params *const params, struct tensor *const t);
static inline void zero_grad(struct model_params *const params);

static inline void zero_grad(struct model_params *const params)
{
    for (size_t i = 0; i < params->size; i++)
    {
        struct tensor *grad = params->params[i]->grad;
        memset(grad->data, 0, grad->data_size * sizeof(double));
    }
}

#endif