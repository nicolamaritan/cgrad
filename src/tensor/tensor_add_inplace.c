#include "tensor/tensor_add_inplace.h"

/**
 * @brief Adds the elements of tensor B to tensor A in place without bounds checking.
 *
 * @param A Pointer to the tensor to which elements will be added.
 * @param B Pointer to the tensor whose elements will be added.
 */
static void tensor_add_inplace_unchecked(struct tensor *A, const struct tensor *const B);
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

    tensor_add_inplace_unchecked(A, B);
    return NO_ERROR;
}

void tensor_add_inplace_unchecked(struct tensor *A, const struct tensor *const B)
{
    switch (A->dtype)
    {
        case DTYPE_FLOAT64:
            tensor_add_inplace_unchecked_f64(A, B);
            break;
        default:
            break;
    }
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