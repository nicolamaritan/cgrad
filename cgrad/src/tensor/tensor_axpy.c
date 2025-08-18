#include "cgrad/tensor/tensor_axpy.h"
#include "cgrad/tensor/tensor_helpers.h"
#include <cblas.h>

static inline cgrad_error tensor_axpy_dispatch(const struct tensor *const x, struct tensor *const y, const double alpha);
static cgrad_error tensor_axpy_f64(const struct tensor *const x, struct tensor *const y, const double alpha);
static cgrad_error tensor_axpy_f32(const struct tensor *const x, struct tensor *const y, const double alpha);

cgrad_error tensor_axpy(const struct tensor *const x, struct tensor *const y, const double alpha)
{
    if (!tensor_same_shape(x, y))
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (x->dtype != y->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor_axpy_dispatch(x, y, alpha);
}

static inline cgrad_error tensor_axpy_dispatch(const struct tensor *const x, struct tensor *const y, const double alpha)
{
    switch (x->dtype)
    {
    case DTYPE_FLOAT64:
        tensor_axpy_f64(x, y, alpha);
        break;
    case DTYPE_FLOAT32:
        tensor_axpy_f32(x, y, alpha);
        break;
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }

    return NO_ERROR;
}

static cgrad_error tensor_axpy_f64(const struct tensor *const x, struct tensor *const y, const double alpha)
{
    const blasint TENSOR_STRIDES = 1;
    cblas_daxpy(
        x->data_size,
        alpha,
        x->data,
        TENSOR_STRIDES,
        y->data,
        TENSOR_STRIDES);

    return NO_ERROR;
}

static cgrad_error tensor_axpy_f32(const struct tensor *const x, struct tensor *const y, const double alpha)
{
    const blasint TENSOR_STRIDES = 1;
    cblas_saxpy(
        x->data_size,
        alpha,
        x->data,
        TENSOR_STRIDES,
        y->data,
        TENSOR_STRIDES);

    return NO_ERROR;
}