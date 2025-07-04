#ifndef RELU_H
#define RELU_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"
#include <stddef.h>

cgrad_error relu_forward_graph(struct tensor *const x, struct tensor *const out);
cgrad_error relu_forward(const struct tensor *const x, struct tensor *const out);

#endif