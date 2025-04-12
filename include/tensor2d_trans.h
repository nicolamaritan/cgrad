#ifndef TENSOR2D_TRANS
#define TENSOR2D_TRANS

#include "tensor.h"
#include "backpropagation_function.h"

typedef enum tensor2d_trans_operand{
    ONLY_OPERAND = 0,
} tensor2d_trans_operand;

tensor_error tensor2d_trans_graph(tensor *const t, tensor *const out);
tensor_error tensor2d_trans(const tensor *const t, tensor *const out);
void tensor2d_trans_unchecked(const tensor *const t, tensor *const out);
void tensor2d_trans_backpropagate(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);

#endif