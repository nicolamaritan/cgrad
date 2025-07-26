#ifndef TENSOR2D_MULT_H
#define TENSOR2D_MULT_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation_function.h"
#include "autograd/autograd_allocators.h"

cgrad_error tensor2d_mult(const struct tensor *const lhs, const struct tensor *const rhs, struct tensor *const out);
cgrad_error tensor2d_mult_graph(struct tensor *const lhs, struct tensor *const rhs, struct tensor *const out, struct autograd_allocators *allocators);

#endif