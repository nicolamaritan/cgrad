#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include "memory/allocators.h"

cgrad_error tensor_add(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
cgrad_error tensor_add_graph(struct tensor *const x, struct tensor *const y, struct tensor *const out, struct allocators *allocs);

#endif