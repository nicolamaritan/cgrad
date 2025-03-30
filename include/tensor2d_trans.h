#ifndef TENSOR2D_TRANS
#define TENSOR2D_TRANS

#include "tensor.h"
#include "backpropagation_function.h"

tensor_error tensor2d_trans_graph(tensor *const t, tensor *const out);
tensor_error tensor2d_trans(const tensor *const t, tensor *const out);
void tensor2d_trans_unchecked(const tensor *const t, tensor *const out);
void tensor2d_trans_backpropagate(const backpropagation_function_data* const data, const tensor* const grad_wrt_out, tensor* grad_wrt_operand, size_t operand);

#endif