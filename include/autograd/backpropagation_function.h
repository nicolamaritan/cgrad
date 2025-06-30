#ifndef BACKPROPAGATION_FUNCTION_H
#define BACKPROPAGATION_FUNCTION_H

#include "tensor/tensor.h"

typedef void (*backpropagation_function)(const struct tensor** const operands, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand);

#endif