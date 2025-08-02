#ifndef TENSOR2D_MULT_H
#define TENSOR2D_MULT_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation_function.h"
#include "memory/allocators.h"

cgrad_error tensor2d_mult(const struct tensor *const lhs, const struct tensor *const rhs, struct tensor *const out);
cgrad_error tensor2d_mult_graph(struct tensor *const lhs, struct tensor *const rhs, struct tensor *const out, struct allocators *allocs);

#endif