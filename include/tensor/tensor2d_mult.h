#ifndef TENSOR2D_MULT_H
#define TENSOR2D_MULT_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"
#include "autograd/autograd_allocators.h"

cgrad_error tensor2d_mult(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);
cgrad_error tensor2d_mult_graph(struct tensor *const A, struct tensor *const B, struct tensor *const out, struct autograd_allocators *allocators);
void tensor2d_mult_unchecked(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);

#endif