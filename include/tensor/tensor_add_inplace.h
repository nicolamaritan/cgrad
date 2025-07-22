#ifndef TENSOR_ADD_INPLACE_H
#define TENSOR_ADD_INPLACE_H

#include "tensor/tensor.h"
#include "utils/error.h"

/**
 * @brief Adds the elements of tensor B to tensor A in place with bounds checking.
 *
 * @param A Pointer to the tensor to which elements will be added.
 * @param B Pointer to the tensor whose elements will be added.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error tensor_add_inplace(struct tensor *A, const struct tensor *const B);

#endif