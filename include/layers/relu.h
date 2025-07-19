#ifndef RELU_H
#define RELU_H

#include "tensor/tensor.h"
#include "autograd/autograd_allocators.h"
#include <stddef.h>

cgrad_error relu_forward_graph(struct tensor *const x, struct tensor *const out, struct autograd_allocators *ag_allocators);
cgrad_error relu_forward(const struct tensor *const x, struct tensor *const out);
void relu_forward_unchecked(const struct tensor *const x, struct tensor *const out);

#endif