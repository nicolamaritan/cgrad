#ifndef TENSOR_HELPER_H
#define TENSOR_HELPER_H

#include "tensor/tensor.h"

/**
 * @brief Prints the contents of the tensor.
 *
 * @param t Pointer to the tensor to be printed.
 */
void print_tensor(const struct tensor *const t);

/**
 * @brief Checks if two tensors have the same shape.
 *
 * @param A Pointer to the first tensor.
 * @param B Pointer to the second tensor.
 * @return True if the tensors have the same shape, otherwise false.
 */
static inline bool tensor_same_shape(const struct tensor *const A, const struct tensor *const B);

/**
 * @brief TODO add documentation
 */
static inline cgrad_error tensor_check_null(const struct tensor *const t);

static inline bool tensor_same_shape(const struct tensor *const A, const struct tensor *const B)
{
    if (A->shape_size != B->shape_size)
    {
        return false;
    }

    size_t shape_size = A->shape_size;
    for (size_t i = 0; i < shape_size; i++)
    {
        if (A->shape[i] != B->shape[i])
        {
            return false;
        }
    }
    return true;
}

static inline cgrad_error tensor_check_null(const struct tensor *const t)
{
    if (t == NULL)
    {
        return TENSOR_NULL;
    }
    if (t->data == NULL)
    {
        return TENSOR_DATA_NULL;
    }
    return NO_ERROR;
}

#endif