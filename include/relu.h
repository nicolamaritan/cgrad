#ifndef RELU_H
#define RELU_H

#include "tensor.h"
#include "computational_graph.h"
#include <stddef.h>

typedef enum relu_layer_operand {
    ONLY_OPERAND = 0,
} relu_layer_operand;

void relu_backpropagate(const tensor ** const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);
tensor_error relu_forward_graph(tensor* const x, tensor* const out);
tensor_error relu_forward(const tensor* const x, tensor* const out);

#endif