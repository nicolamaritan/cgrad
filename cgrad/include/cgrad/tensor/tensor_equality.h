#ifndef TENSOR_EQUALITY_H
#define TENSOR_EQUALITY_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/config.h"
#include <stdbool.h>

bool tensor_no_grad_equal(const struct tensor *const t1, const struct tensor *const t2);
bool tensor_equal(const struct tensor *const t1, const struct tensor *const t2);

/**
 * @brief Checks if two tensors have the same shape.
 *
 * @param t1 Pointer to the first tensor.
 * @param t2 Pointer to the second tensor.
 * @return True if the tensors have the same shape, otherwise false.
 */
bool tensor_same_shape(const struct tensor *const t1, const struct tensor *const t2);

bool tensor_no_grad_same_data(const struct tensor *const t1, const struct tensor *const t2);

#endif