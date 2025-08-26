#ifndef TENSOR_HELPER_H
#define TENSOR_HELPER_H

#include "cgrad/tensor/tensor.h"

/**
 * @brief Prints the contents of the tensor.
 *
 * @param t Pointer to the tensor to be printed.
 */
void print_tensor(const struct tensor *const t);


/**
 * @brief TODO add documentation
 */
static inline cgrad_error tensor_check_null(const struct tensor *const t);

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