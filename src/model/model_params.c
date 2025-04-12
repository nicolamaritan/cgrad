#include "model/model_params.h"

int add_param(model_params* const params, tensor* const t)
{
    size_t const size = params->size;
    if (size >= MAX_PARAMS)
    {
        return 1;
    }

    params->params[size] = t;
    params->size++;

    return 0;
}