#include "cgrad/tensor/tensor_scalar_mult_tensor_add.h"
#include "cgrad/tensor/tensor_helpers.h"
#include <cblas.h>

static inline cgrad_error tensor_scalar_mult_tensor_add_dispatch(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out);
static cgrad_error tensor_scalar_mult_tensor_add_f64(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out);
static cgrad_error tensor_scalar_mult_tensor_add_f32(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out);

cgrad_error tensor_scalar_mult_tensor_add(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out)
{
    if (!x || !y)
    {
        return TENSOR_NULL;
    }
    if (!tensor_same_shape(x, y))
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (x->dtype != y->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor_scalar_mult_tensor_add_dispatch(x, y, alpha, out);
}

static inline cgrad_error tensor_scalar_mult_tensor_add_dispatch(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out)
{
    switch (x->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor_scalar_mult_tensor_add_f64(x, y, alpha, out);
    case DTYPE_FLOAT32:
        return tensor_scalar_mult_tensor_add_f32(x, y, alpha, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_scalar_mult_tensor_add_f64(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out)
{
    double *x_data = (double *)x->data;
    double *y_data = (double *)y->data;
    double *out_data = (double *)out->data;

    for (size_t j = 0; j < x->data_size; j++)
    {
        out_data[j] = alpha * x_data[j] + y_data[j];
    }

    return NO_ERROR;
}

static cgrad_error tensor_scalar_mult_tensor_add_f32(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out)
{
    float *x_data = (float *)x->data;
    float *y_data = (float *)y->data;
    float *out_data = (float *)out->data;

    for (size_t j = 0; j < x->data_size; j++)
    {
        out_data[j] = alpha * x_data[j] + y_data[j];
    }

    return NO_ERROR;
}