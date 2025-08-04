#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include "memory/allocators.h"

cgrad_error tensor_add(struct tensor *const x, struct tensor *const y, struct tensor **const out, const bool track_grad, struct allocators *const allocs);

#endif