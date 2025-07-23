#include "tensor/tensor_scalar_mult_tensor_sum.h"
#include <cblas.h>

static cgrad_error tensor_scalar_mult_tensor_sum_dispatch(struct tensor *const A, struct tensor *const B, const double alpha, struct tensor *const out);
static void tensor_scalar_mult_tensor_sum_unchecked_f64(struct tensor *const A, struct tensor *const B, const double alpha, struct tensor *const out);

cgrad_error tensor_scalar_mult_tensor_sum(struct tensor *const A, struct tensor *const B, const double alpha, struct tensor *const out)
{
    if (!A || !B)
    {
        return TENSOR_NULL;
    }
    if (!tensor_same_shape(A, B))
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (A->dtype != B->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor_scalar_mult_tensor_sum_dispatch(A, B, alpha, out);
}

static cgrad_error tensor_scalar_mult_tensor_sum_dispatch(struct tensor *const A, struct tensor *const B, const double alpha, struct tensor *const out)
{
    switch (A->dtype)
    {
    case DTYPE_FLOAT64:
        tensor_scalar_mult_tensor_sum_unchecked_f64(A, B, alpha, out);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void tensor_scalar_mult_tensor_sum_unchecked_f64(struct tensor *const A, struct tensor *const B, const double alpha, struct tensor *const out)
{
    double *A_data = (double *)A->data;
    double *B_data = (double *)B->data;
    double *out_data = (double *)out->data;

    for (size_t j = 0; j < A->data_size; j++)
    {
        out_data[j] = alpha * A_data[j] + B_data[j];
    }
}