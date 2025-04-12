#include "tensor/tensor2d_trans.h"
#include "autograd/computational_graph.h"

tensor_error tensor2d_trans(const tensor *const t, tensor *const out)
{
    if (!t || !out)
        return TENSOR_NULL;
    if (!t->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (!t->data || !out->data)
        return TENSOR_DATA_NULL;
    if (t->shape[0] != out->shape[1] || t->shape[1] != out->shape[0])
        return TENSOR_SHAPE_MISMATCH;

    tensor2d_trans_unchecked(t, out);
    return TENSOR_OK;
}

tensor_error tensor2d_trans_graph(tensor *const t, tensor *const out)
{
    tensor_error error = tensor2d_trans(t, out);

    if (error != TENSOR_OK)
        return error;

    // Update computational graph
    computational_graph_node *t_node = t->node ? t->node : computational_graph_node_tensor_alloc(t);
    computational_graph_node *out_node = computational_graph_node_tensor_alloc(out);

    // Unused operand
    add_parent(t_node, out_node, TENSOR2D_TRANS_ONLY_OPERAND);
    add_child(out_node, t_node);

    // Setup backpropagation function
    out_node->function[TENSOR2D_TRANS_ONLY_OPERAND] = (backpropagation_function)&tensor2d_trans_backpropagate;

    // Setup operands
    out_node->tensor_operands[TENSOR2D_TRANS_ONLY_OPERAND] = t;

    return TENSOR_OK;
}

void tensor2d_trans_unchecked(const tensor *const t, tensor *const out)
{
    // Extract the shape of t 
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    out->shape[0] = cols;
    out->shape[1] = rows;

    // Transpose
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            out->data[j * rows + i] = t->data[i * cols + j];
        }
    }
}
void tensor2d_trans_backpropagate(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand)
{
    tensor2d_trans(grad_wrt_out, grad_wrt_operand);
}