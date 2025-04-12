#include "tensor/tensor2d_add_row_vector.h"
#include "autograd/computational_graph.h"
#include <stdlib.h>

void tensor2d_add_row_vector_unchecked(const tensor *const A, const tensor *const v, tensor *out);

tensor_error tensor2d_add_row_vector(const tensor *const A, const tensor *const v, tensor *const out)
{
    if (!A || !v)
        return TENSOR_NULL;
    if (!A->data || !v->data)
        return TENSOR_DATA_NULL;
    if (!A->shape || !v->shape)
        return TENSOR_SHAPE_NULL;
    if (A->shape_size != 2 || v->shape_size != 2)
        return TENSOR_WRONG_SHAPE;
    if (v->shape[1] != 1)
        return TENSOR_WRONG_SHAPE;
    if (A->shape[1] != v->shape[0])
        return TENSOR_SHAPE_MISMATCH;

    tensor2d_add_row_vector_unchecked(A, v, out);

    return TENSOR_OK;
}

tensor_error tensor2d_add_row_vector_graph(tensor *const A, tensor *const v, tensor *const out)
{
    if (!A || !v)
        return TENSOR_NULL;
    if (!A->data || !v->data)
        return TENSOR_DATA_NULL;
    if (!A->shape || !v->shape)
        return TENSOR_SHAPE_NULL;
    if (A->shape_size != 2 || v->shape_size != 2)
        return TENSOR_WRONG_SHAPE;
    if (v->shape[1] != 1)
        return TENSOR_WRONG_SHAPE;
    if (A->shape[1] != v->shape[0])
        return TENSOR_SHAPE_MISMATCH;

    tensor2d_add_row_vector_unchecked(A, v, out);

    computational_graph_node *A_node = A->node ? A->node : computational_graph_node_tensor_alloc(A);
    computational_graph_node *v_node = v->node ? v->node : computational_graph_node_tensor_alloc(v);
    computational_graph_node *out_node = computational_graph_node_tensor_alloc(out);

    // Setup connections
    add_parent(A_node, out_node, TENSOR2D);
    add_parent(v_node, out_node, ROW_VECTOR);
    add_child(out_node, A_node);
    add_child(out_node, v_node);

    // Setup backpropagation functions
    out_node->function[TENSOR2D] = (backpropagation_function)&tensor2d_add_row_vector_backpropagate_tensor2d;
    out_node->function[ROW_VECTOR] = (backpropagation_function)&tensor2d_add_row_vector_backpropagate_row_vector;

    // Setup operands
    out_node->tensor_operands[TENSOR2D] = A;
    out_node->tensor_operands[ROW_VECTOR] = v;

    return TENSOR_OK;
}

void tensor2d_add_row_vector_unchecked(const tensor *const A, const tensor *const v, tensor *out)
{
    size_t rows = A->shape[0]; // Number of rows
    size_t cols = A->shape[1]; // Number of columns

    double *A_data = A->data;
    double *v_data = v->data;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            // Unchecked is invoked for performance reasons.
            tensor2d_set_unchecked(out, i, j, A_data[i * cols + j] + v_data[j]);
        }
    }
}

void tensor2d_add_row_vector_backpropagate_tensor2d(const tensor **const tensor_operands, const tensor *const grad_wrt_out, tensor *grad_wrt_operand)
{
    tensor2d_copy(grad_wrt_out, grad_wrt_operand);
}

void tensor2d_add_row_vector_backpropagate_row_vector(const tensor **const operands, const tensor *const grad_wrt_out, tensor *grad_wrt_operand)
{
    size_t G_rows = grad_wrt_out->shape[0];
    size_t G_cols = grad_wrt_out->shape[1];

    for (size_t j = 0; j < G_cols; j++)
        grad_wrt_operand->data[j] = 0;

    // Iterating by row since vectors are stored in row-major
    for (size_t i = 0; i < G_rows; i++)
        for (size_t j = 0; j < G_cols; j++)
            grad_wrt_operand->data[j] += grad_wrt_out->data[i * G_cols + j];
}