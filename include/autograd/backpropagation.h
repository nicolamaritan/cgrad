#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "tensor/tensor.h"
#include "autograd/autograd_allocators.h"
#include "memory/tensor_allocator.h"
#include "memory/computational_graph_allocator.h"
#include <stdbool.h>

void backward(struct tensor* t, struct autograd_allocators *allocators);

#endif