#ifndef TENSOR_COPY_H
#define TENSOR_COPY_H

#include "tensor/tensor.h"

/**
 * @brief Copies the contents of a tensor from source to destination with bounds checking.
 *
 * @param src Pointer to the source tensor.
 * @param dest Pointer to the destination tensor.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error tensor_copy(const struct tensor *const src, struct tensor *const dest);

/**
 * @brief Copies the contents of a 2D tensor from source to destination with bounds checking.
 *
 * @param src Pointer to the source tensor.
 * @param dest Pointer to the destination tensor.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error tensor2d_copy(const struct tensor *const src, struct tensor *const dest);

#endif