#include "tensor/tensor_copy.h"
#include <string.h>

cgrad_error tensor_copy(const struct tensor *const src, struct tensor *const dest)
{
    if (!src || !dest)
    {
        return TENSOR_NULL;
    }
    if (!src->data || !dest->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (src->shape_size != dest->shape_size)
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    for (size_t i = 0; i < src->shape_size; i++)
    {
        if (src->shape[i] != dest->shape[i])
        {
            return TENSOR_SHAPE_MISMATCH;
        }
    }

    memcpy(dest->data, src->data, sizeof(double) * src->data_size);

    return NO_ERROR;
}

cgrad_error tensor2d_copy(const struct tensor *const src, struct tensor *const dest)
{
    if (!src || !dest)
    {
        return TENSOR_NULL;
    }
    if (!src->data || !dest->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (src->shape[0] != dest->shape[0] || src->shape[1] != dest->shape[1])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
        
    memcpy(dest->data, src->data, src->shape[0] * src->shape[1] * dtype_sizeof(src->dtype));
    return NO_ERROR;
}
