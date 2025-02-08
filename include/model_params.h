#ifndef MODEL_PARAMS
#define MODEL_PARAMS

#include "tensor.h"

#define MAX_PARAMS 1024

typedef struct 
{
    tensor* params[MAX_PARAMS];
    size_t size;
} model_params;

int add_param(model_params* const params, tensor* const t);

#endif