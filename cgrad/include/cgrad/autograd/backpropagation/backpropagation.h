#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/memory/allocators.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_allocator.h"
#include "cgrad/error.h"
#include <stdbool.h>

cgrad_error backward(struct tensor* t, struct allocators *allocs);

#endif