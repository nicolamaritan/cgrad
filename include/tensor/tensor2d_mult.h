#ifndef TENSOR2D_MULT_H
#define TENSOR2D_MULT_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"

typedef enum tensor2d_mult_operand
{
    LHS_TENSOR = 0,
    RHS_TENSOR = 1,
} tensor2d_mult_operand;

cgrad_error tensor2d_mult(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);
cgrad_error tensor2d_mult_graph(struct tensor *const A, struct tensor *const B, struct tensor *const out);
void tensor2d_mult_backpropagate_lhs(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
void tensor2d_mult_backpropagate_rhs(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

#endif