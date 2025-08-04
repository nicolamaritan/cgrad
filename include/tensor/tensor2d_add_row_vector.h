#ifndef TENSOR2D_ADD_ROW_VECTOR_H
#define TENSOR2D_ADD_ROW_VECTOR_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation_function.h"
#include "memory/allocators.h"

cgrad_error tensor2d_add_row_vector(struct tensor *const t, struct tensor *const v, struct tensor **const out, const bool track_grad, struct allocators *const allocs);
cgrad_error tensor2d_add_row_vector_into(const struct tensor *const t, const struct tensor *const v, struct tensor *const out);

#endif