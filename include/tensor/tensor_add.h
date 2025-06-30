#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"

typedef enum tensor_add_operand
{
    LHS_TENSOR = 0,
    RHS_TENSOR = 1,
} tensor_add_operand;

cgrad_error tensor_add(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);
cgrad_error tensor_add_graph(struct tensor *const A, struct tensor *const B, struct tensor *const out);
void tensor_add_unchecked(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);
void tensor_add_backpropagate(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);


#endif