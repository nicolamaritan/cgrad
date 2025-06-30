#ifndef TENSOR2D_TRANS_H
#define TENSOR2D_TRANS_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"

typedef enum tensor2d_trans_operand{
    TENSOR2D_TRANS_ONLY_OPERAND = 0,
} tensor2d_trans_operand;

cgrad_error tensor2d_trans_graph(struct tensor *const t, struct tensor *const out);
cgrad_error tensor2d_trans(const struct tensor *const t, struct tensor *const out);
void tensor2d_trans_unchecked(const struct tensor *const t, struct tensor *const out);
void tensor2d_trans_backpropagate(const struct tensor **const operands, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand);

#endif