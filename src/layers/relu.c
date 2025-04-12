#include "layers/relu.h"
#include <stdlib.h>
#include <stdio.h>

void relu_backpropagate(const tensor **const operands, const tensor* const grad_wrt_out, tensor *grad_wrt_operand)
{
    const tensor *const x = operands[RELU_ONLY_OPERAND];
    
    // Avoid multiple indirections for performance
    double* x_data = x->data;
    double* grad_wrt_operand_data = grad_wrt_operand->data;
    size_t grad_wrt_operand_data_size = grad_wrt_operand->data_size;
    
    /*
        Gradient computation of dz/dX.
        dz/dX is the Hadamard Product of grad_wrt_out = dz/drelu(X) and drelu(X)/dX,
        since element (i, j) of relu(X) depends only on element (i, j) of X.
    */
    
    for (size_t i = 0; i < grad_wrt_operand_data_size; i++)
    {
        // Element wise product
        grad_wrt_operand_data[i] = (x_data[i] > 0 ? 1 : 0) * grad_wrt_out->data[i];
    }
}

tensor_error relu_forward_graph(tensor* const x, tensor* const out)
{
    tensor_error error = relu_forward(x, out);
    if (error != TENSOR_OK)
        return error;

    computational_graph_node* x_node = x->node ? x->node : computational_graph_node_tensor_alloc(x);
    computational_graph_node* out_node = computational_graph_node_tensor_alloc(out);

    add_parent(x_node, out_node, RELU_ONLY_OPERAND);
    add_child(out_node, x_node);

    // Setup backpropation functions 
    out_node->function[RELU_ONLY_OPERAND] = (backpropagation_function)&relu_backpropagate;

    // Setup operands
    out_node->tensor_operands[RELU_ONLY_OPERAND] = x;

    return TENSOR_OK;
}

tensor_error relu_forward(const tensor* const x, tensor* const out)
{
    if (!x || !out)
        return TENSOR_NULL;
    if (!x->data || !out->data)
        return TENSOR_DATA_NULL;
    if (!x->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (!tensor_same_shape(x, out))
        return TENSOR_SHAPE_MISMATCH;

    // Avoid multiple indirections for performance
    double* x_data = x->data;
    double* out_data = out->data;
    size_t out_data_size = out->data_size;

    for (size_t i = 0; i < out_data_size; i++)
    {
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }

    return TENSOR_OK;
}