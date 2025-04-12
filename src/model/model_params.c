#include "model/model_params.h"

cgrad_error add_param(model_params* const params, tensor* const t)
{
    size_t const size = params->size;
    if (size >= MODEL_MAX_PARAMS)
    {
        return MODEL_MAX_PARAMS_EXCEEDED;
    }

    params->params[size] = t;
    params->size++;

    return NO_ERROR;
}