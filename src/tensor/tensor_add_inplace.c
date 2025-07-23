#include "tensor/tensor_add_inplace.h"

static cgrad_error tensor_add_inplace_dispatch(struct tensor *A, const struct tensor *const B);
static void tensor_add_inplace_unchecked_f64(struct tensor *A, const struct tensor *const B);

cgrad_error tensor_add_inplace(struct tensor *A, const struct tensor *const B)
{
    if (!A || !B)
    {
        return TENSOR_NULL;
    }
    if (!A->data || !B->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (A->data_size != B->data_size)
    {
        return TENSOR_DATA_SIZE_MISMATCH;
    }
    if (!tensor_same_shape(A, B))
    {
        return false;
    }

    return tensor_add_inplace_dispatch(A, B);
}

static cgrad_error tensor_add_inplace_dispatch(struct tensor *A, const struct tensor *const B)
{
    switch (A->dtype)
    {
    case DTYPE_FLOAT64:
        tensor_add_inplace_unchecked_f64(A, B);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

void tensor_add_inplace_unchecked_f64(struct tensor *A, const struct tensor *const B)
{
    double *A_data = (double *)A->data;
    double *B_data = (double *)B->data;

    for (size_t i = 0; i < A->data_size; i++)
    {
        A_data[i] += B_data[i];
    }
}