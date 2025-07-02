#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"

cgrad_error tensor_add(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);
cgrad_error tensor_add_graph(struct tensor *const A, struct tensor *const B, struct tensor *const out);
void tensor_add_unchecked(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);

#endif