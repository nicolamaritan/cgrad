#include "tensor/tensor_add.h"
#include "autograd/computational_graph.h"

void tensor_add_unchecked(const tensor *const A, const tensor *const B, tensor *const out)
{
    for (size_t i = 0; i < A->data_size; i++)
        out->data[i] = A->data[i] + B->data[i];
}

cgrad_error tensor_add(const tensor *const A, const tensor *const B, tensor *const out)
{
    if (!A || !B || !out)
        return TENSOR_NULL;
    if (!A->data || !B->data || !out->data)
        return TENSOR_DATA_NULL;
    if (!A->shape || !B->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (A->data_size != B->data_size || B->data_size != out->data_size)
        return TENSOR_DATA_SIZE_MISMATCH;
    if (!tensor_same_shape(A, B))
        return TENSOR_SHAPE_MISMATCH;

    tensor_add_unchecked(A, B, out);
    return NO_ERROR;
}

cgrad_error tensor_add_graph(tensor *const A, tensor *const B, tensor *const out)
{
    cgrad_error err = tensor_add(A, B, out);
    if (err != NO_ERROR)
    {
        return err;
    }

    // Update computational graph
    err = add_computational_graph_link(A, LHS_TENSOR, out, &tensor_add_backpropagate);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(B, RHS_TENSOR, out, &tensor_add_backpropagate);

    return err;
}

void tensor_add_backpropagate(const tensor **const operands, const tensor *const grad_wrt_out, tensor *grad_wrt_operand)
{
    /**
     * Given the symmetry of the addition operation, the gradient with respect to both operands is the same.
     * Therefore, we can use the same gradient for both operands.
     * The gradient with respect to both operands is the gradient with respect to the output. 
     */
    tensor_copy(grad_wrt_out, grad_wrt_operand);
}