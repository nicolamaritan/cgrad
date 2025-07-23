#include "tensor/tensor_axpy.h"
#include <cblas.h>

static void tensor_axpy_unchecked(struct tensor *const X, struct tensor *const Y, const double alpha);
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

    tensor_axpy_unchecked(X, Y, alpha);
    return NO_ERROR;
}

static void tensor_axpy_unchecked(struct tensor *const X, struct tensor *const Y, const double alpha)
{
    switch (X->dtype)
    {
    case DTYPE_FLOAT64:
        tensor_axpy_unchecked_f64(X, Y, alpha);
        break;
    default:
        break;
    }
}

static void tensor_axpy_unchecked_f64(struct tensor *const X, struct tensor *const Y, const double alpha)
{
    cblas_daxpy(
        X->data_size,
        alpha,
        X->data,
        1,
        Y->data,
        1
    );
}