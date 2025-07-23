#include "tensor/tensor_axpy.h"
#include <cblas.h>

static cgrad_error tensor_axpy_dispatch(struct tensor *const X, struct tensor *const Y, const double alpha);
static void tensor_axpy_unchecked_f64(struct tensor *const X, struct tensor *const Y, const double alpha);

cgrad_error tensor_axpy(struct tensor *const X, struct tensor *const Y, const double alpha)
{
    if (!tensor_same_shape(X, Y))
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (X->dtype != Y->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor_axpy_dispatch(X, Y, alpha);
}

static cgrad_error tensor_axpy_dispatch(struct tensor *const X, struct tensor *const Y, const double alpha)
{
    switch (X->dtype)
    {
    case DTYPE_FLOAT64:
        tensor_axpy_unchecked_f64(X, Y, alpha);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void tensor_axpy_unchecked_f64(struct tensor *const X, struct tensor *const Y, const double alpha)
{
    const blasint TENSOR_STRIDES = 1;
    cblas_daxpy(
        X->data_size,
        alpha,
        X->data,
        TENSOR_STRIDES,
        Y->data,
        TENSOR_STRIDES);
}