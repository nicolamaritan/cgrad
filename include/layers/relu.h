#ifndef RELU_H
#define RELU_H

#include "tensor/tensor.h"
#include "memory/allocators.h"
#include <stddef.h>

cgrad_error relu_forward(const struct tensor *const x, struct tensor **const out, struct allocators *allocs);
cgrad_error relu_forward_graph(struct tensor *const x, struct tensor **const out, struct allocators *allocs);

#endif