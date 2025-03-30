#ifndef TENSOR2D_MULT
#define TENSOR2D_MULT

#include "tensor.h"
#include "backpropagation_function.h"

typedef enum tensor2d_mult_operand{
    LHS_TENSOR = 0,
    RHS_TENSOR = 1,
} tensor2d_mult_operand;

typedef struct
{
    tensor* lhs_tensor;
    tensor* rhs_tensor;
} tensor2d_mult_inputs;

tensor_error tensor2d_mult(const tensor *const A, const tensor *const B, tensor *const out);
tensor_error tensor2d_mult_graph(tensor *const A, tensor *const B, tensor *const out);
void tensor2d_mult_backpropagate(const backpropagation_function_data* const data, const tensor* const grad_wrt_out, tensor* grad_wrt_operand, size_t operand);
void free_tensor2d_mult_backpropagation_function_data(backpropagation_function_data* data);

#endif