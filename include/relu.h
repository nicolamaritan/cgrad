#ifndef RELU_H
#define RELU_H

#include "tensor.h"
#include "computational_graph.h"
#include <stddef.h>

typedef enum relu_layer_operand {
    INPUT = 0,
} relu_layer_operand;

tensor* relu_backpropagate(const backpropagation_function_data* const data, const tensor* const D, size_t operand);
void relu_forward_graph(tensor* const x, tensor* const out);
void relu_forward(const tensor* const x, tensor* const out);
void free_relu_backpropagation_function_data(backpropagation_function_data* data);

#endif