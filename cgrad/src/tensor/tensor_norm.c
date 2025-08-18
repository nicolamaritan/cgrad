#include "cgrad/tensor/tensor_norm.h"

static inline cgrad_error tensor_norm_dispatch(const struct tensor *const t, double *const out);
static inline cgrad_error tensor_norm_f32(const struct tensor *const t, double *const out);

static inline cgrad_error tensor_norm(const struct tensor *const t, double *const out)
{
    if (!t)
    {
        return TENSOR_NULL;
    }

    return tensor_norm_dispatch(t, out);
}

static inline cgrad_error tensor_norm_dispatch(const struct tensor *const t, double *const out)
{
    switch (t->dtype)
    {
        case DTYPE_FLOAT32:
            return tensor_norm_f32(t, out);
        default:
            return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static inline cgrad_error tensor_norm_f32(const struct tensor *const t, double *const out)
{
    *out = 0.0f;
    float* t_data = (float*)t->data;
    for (size_t i = 0;  i < t->data_size; i++)
    {
        *out += (t_data[i] * t_data[i]);
    }

    return NO_ERROR;
}