#ifndef RELU_H
#define RELU_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"
#include <stddef.h>

typedef enum relu_layer_operand
{
    RELU_ONLY_OPERAND = 0,
} relu_layer_operand;

void relu_backpropagate(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
cgrad_error relu_forward_graph(struct tensor *const x, struct tensor *const out);
cgrad_error relu_forward(const struct tensor *const x, struct tensor *const out);

#endif