#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"

typedef enum tensor_add_operand
{
    LHS_TENSOR = 0,
    RHS_TENSOR = 1,
} tensor_add_operand;

tensor_error tensor_add(const tensor *const A, const tensor *const B, tensor *const out);
tensor_error tensor_add_graph(tensor *const A, tensor *const B, tensor *const out);
void tensor_add_unchecked(const tensor *const A, const tensor *const B, tensor *const out);
void tensor_add_backpropagate(const tensor **const operands, const tensor *const grad_wrt_out, tensor *grad_wrt_operand);


#endif