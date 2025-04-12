#ifndef BACKPROPAGATION_FUNCTION_H
#define BACKPROPAGATION_FUNCTION_H

#include "tensor/tensor.h"

typedef void (*backpropagation_function)(const tensor** const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);

#endif