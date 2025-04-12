#ifndef TENSOR2D_MULT_H
#define TENSOR2D_MULT_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"

typedef enum tensor2d_mult_operand{
    LHS_TENSOR = 0,
    RHS_TENSOR = 1,
} tensor2d_mult_operand;

cgrad_error tensor2d_mult(const tensor *const A, const tensor *const B, tensor *const out);
cgrad_error tensor2d_mult_graph(tensor *const A, tensor *const B, tensor *const out);
void tensor2d_mult_backpropagate_lhs(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);
void tensor2d_mult_backpropagate_rhs(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);

#endif