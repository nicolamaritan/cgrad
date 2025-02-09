#ifndef MODEL_PARAMS
#define MODEL_PARAMS

#include "tensor.h"
#include <string.h>

#define MAX_PARAMS 1024

typedef struct 
{
    tensor* params[MAX_PARAMS];
    size_t size;
} model_params;

int add_param(model_params* const params, tensor* const t);
static inline void zero_grad(model_params* const params);

static inline void zero_grad(model_params* const params)
{
    for (size_t i = 0; i < params->size; i++)
    {
        tensor* grad = params->params[i]->grad;
        memset(grad->data, 0, grad->data_size * sizeof(double));
    }
}

#endif