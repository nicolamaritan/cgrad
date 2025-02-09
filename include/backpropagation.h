#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "tensor.h"
#include <stdbool.h>

void backward(tensor* t, bool retain_graph);
// void backpropagate(backpropagation_targets* const targets);
// void zero_grad(tensor* const root);

#endif