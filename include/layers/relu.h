#ifndef RELU_H
#define RELU_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"
#include <stddef.h>

typedef enum relu_layer_operand {
    RELU_ONLY_OPERAND = 0,
} relu_layer_operand;

void relu_backpropagate(const tensor ** const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);
cgrad_error relu_forward_graph(tensor* const x, tensor* const out);
cgrad_error relu_forward(const tensor* const x, tensor* const out);

#endif