#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "tensor/tensor.h"
#include "config.h"
#include <string.h>

typedef struct
{
    tensor *params[MODEL_MAX_PARAMS];
    size_t size;
} model_params;

void init_model_params(model_params *const params);
cgrad_error add_model_param(model_params *const params, tensor *const t);
static inline void zero_grad(model_params *const params);

static inline void zero_grad(model_params *const params)
{
    for (size_t i = 0; i < params->size; i++)
    {
        tensor *grad = params->params[i]->grad;
        memset(grad->data, 0, grad->data_size * sizeof(double));
    }
}

#endif