#ifndef TENSOR2D_MULT_H
#define TENSOR2D_MULT_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation_function.h"
#include "memory/allocators.h"

cgrad_error tensor2d_mult(struct tensor *const lhs, struct tensor *const rhs, struct tensor **const out, const bool track_grad, struct allocators *const allocs);
cgrad_error tensor2d_mult_into(const struct tensor *const lhs, const struct tensor *const rhs, struct tensor *const out);

#endif