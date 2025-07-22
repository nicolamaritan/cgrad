#ifndef TENSOR2D_ADD_ROW_VECTOR_H
#define TENSOR2D_ADD_ROW_VECTOR_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation_function.h"
#include "autograd/autograd_allocators.h"

cgrad_error tensor2d_add_row_vector_graph(struct tensor *const A, struct tensor *const v, struct tensor *const out, struct autograd_allocators *allocators);
cgrad_error tensor2d_add_row_vector(const struct tensor *const A, const struct tensor *const v, struct tensor *const out);

#endif