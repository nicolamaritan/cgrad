#ifndef TENSOR2D_ADD_ROW_VECTOR
#define TENSOR2D_ADD_ROW_VECTOR

#include "tensor.h"
#include "backpropagation_function.h"

typedef enum tensor2d_add_row_vector_operand{
    TENSOR2D = 0,
    ROW_VECTOR = 1,
} tensor2d_add_row_vector_operand;

typedef struct
{
    tensor* t2d;
    tensor* row_vector;
} tensor2d_add_row_vector_inputs;

tensor_error tensor2d_add_row_vector_graph(tensor *const A, tensor *const v, tensor* const out);
tensor_error tensor2d_add_row_vector(const tensor *const A, const tensor *const v, tensor* const out);
void tensor2d_add_row_vector_backpropagate(const backpropagation_function_data* const data, const tensor* const grad_wrt_out, tensor* grad_wrt_operand, size_t operand);

#endif