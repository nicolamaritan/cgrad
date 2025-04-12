#ifndef TENSOR2D_TRANS_H
#define TENSOR2D_TRANS_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"

typedef enum tensor2d_trans_operand{
    TENSOR2D_TRANS_ONLY_OPERAND = 0,
} tensor2d_trans_operand;

cgrad_error tensor2d_trans_graph(tensor *const t, tensor *const out);
cgrad_error tensor2d_trans(const tensor *const t, tensor *const out);
void tensor2d_trans_unchecked(const tensor *const t, tensor *const out);
void tensor2d_trans_backpropagate(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);

#endif