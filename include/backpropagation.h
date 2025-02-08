#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "tensor.h"

void backward(tensor* t);
// void backpropagate(backpropagation_targets* const targets);
void zero_grad(tensor* const root);

#endif