#include "model/model_params.h"

void model_params_init(struct model_params *const params)
{
    params->size = 0;
    memset(params->params, 0, params->size * sizeof(struct tensor *));
}

cgrad_error add_model_param(struct model_params *const params, struct tensor *const t)
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