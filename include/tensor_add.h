#ifndef TENSOR_ADD
#define TENSOR_ADD

#include "tensor.h"
#include "backpropagation_function.h"

const int UNUSED_OPERAND_VALUE = -1;

tensor_error tensor_add(const tensor *const A, const tensor *const B, tensor *const out);
tensor_error tensor_add_graph(tensor *const A, tensor *const B, tensor *const out);
void tensor_add_unchecked(const tensor *const A, const tensor *const B, tensor *const out);
void tensor_add_backpropagate(const backpropagation_function_data* const data, const tensor* const grad_wrt_out, tensor* grad_wrt_operand, size_t operand);

#endif