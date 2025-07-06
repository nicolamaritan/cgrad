#include "tensor/tensor_add.h"
#include "autograd/computational_graph.h"

typedef enum tensor_add_operand
{
    LHS_TENSOR,
    RHS_TENSOR,
} tensor_add_operand;

static void tensor_add_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

void tensor_add_unchecked(const struct tensor *const A, const struct tensor *const B, struct tensor *const out)
{
    for (size_t i = 0; i < A->data_size; i++)
    {
        out->data[i] = A->data[i] + B->data[i];
    }
}

cgrad_error tensor_add(const struct tensor *const A, const struct tensor *const B, struct tensor *const out)
{
    if (!A || !B || !out)
    {
        return TENSOR_NULL;
    }
    if (!A->data || !B->data || !out->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (A->data_size != B->data_size || B->data_size != out->data_size)
    {
        return TENSOR_DATA_SIZE_MISMATCH;
    }
    if (!tensor_same_shape(A, B))
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    tensor_add_unchecked(A, B, out);
    return NO_ERROR;
}

cgrad_error tensor_add_graph(struct tensor *const A, struct tensor *const B, struct tensor *const out)
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

static void tensor_add_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    /**
     * Given the symmetry of the addition operation, the gradient with respect to both operands is the same.
     * Therefore, we can use the same gradient for both operands.
     * The gradient with respect to both operands is the gradient with respect to the output.
     */
    tensor_copy(grad_wrt_out, grad_wrt_operand);
}