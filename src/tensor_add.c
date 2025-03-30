#include "tensor_add.h"
#include "computational_graph.h"

void tensor_add_unchecked(const tensor *const A, const tensor *const B, tensor *const out)
{
    for (size_t i = 0; i < A->data_size; i++)
        out->data[i] = A->data[i] + B->data[i];
}

tensor_error tensor_add(const tensor *const A, const tensor *const B, tensor *const out)
{
    if (!A || !B || !out)
        return TENSOR_NULL;
    if (!A->data || !B->data || !out->data)
        return TENSOR_DATA_NULL;
    if (!A->shape || !B->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (A->data_size != B->data_size || B->data_size != out->data_size)
        return TENSOR_DATA_SIZE_MISMATCH;
    if (tensor_same_shape(A, B))
        return TENSOR_SHAPE_MISMATCH;

    tensor_add_unchecked(A, B, out);
    return TENSOR_OK;
}

tensor_error tensor_add_graph(tensor *const A, tensor *const B, tensor *const out)
{
    tensor_error err = tensor_add(A, B, out);
    if (err != TENSOR_OK)
        return err;

    computational_graph_node *A_node = A->node ? A->node : computational_graph_node_tensor_alloc(A);
    computational_graph_node *B_node = B->node ? B->node : computational_graph_node_tensor_alloc(B);
    computational_graph_node *out_node = computational_graph_node_tensor_alloc(out);

    // Setup connections
    add_parent(A_node, out_node, UNUSED_OPERAND_VALUE);
    add_parent(B_node, out_node, UNUSED_OPERAND_VALUE);
    add_child(out_node, A_node);
    add_child(out_node, B_node);

    // backpropagation_function function = 
    out_node->function = (backpropagation_function)&tensor_add_backpropagate;
    out_node->data = NULL;
    out_node->free_data = NULL;

    return TENSOR_OK;
}

void tensor_add_backpropagate(const backpropagation_function_data *const data, const tensor *const grad_wrt_out, tensor *grad_wrt_operand, size_t operand)
{
    /**
     * Given the symmetry of the addition operation, the gradient with respect to both operands is the same.
     * Therefore, we can use the same gradient for both operands.
     * The gradient with respect to both operands is the gradient with respect to the output. 
     */
    tensor_copy(grad_wrt_out, grad_wrt_operand);
}