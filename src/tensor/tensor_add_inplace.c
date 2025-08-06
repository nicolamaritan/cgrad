#include "tensor/tensor_add_inplace.h"
#include "tensor/tensor_axpy.h"
#include "tensor/tensor_helpers.h"

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

    return tensor_axpy(B, A, 1.0);
}