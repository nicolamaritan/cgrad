#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "tensor/tensor.h"
#include <stdbool.h>

void backward(struct tensor* t, bool retain_graph);

#endif