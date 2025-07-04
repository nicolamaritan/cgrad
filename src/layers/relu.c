#include "layers/relu.h"
#include <stdlib.h>
#include <stdio.h>

typedef enum relu_layer_operand
{
    RELU_ONLY_OPERAND,
} relu_layer_operand;

static void relu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error relu_forward_graph(struct tensor* const x, struct tensor* const out)
{
    cgrad_error error = relu_forward(x, out);
    if (error != NO_ERROR)
    {
        return error;
    }

    error = add_computational_graph_link(x, RELU_ONLY_OPERAND, out, &relu_backpropagate);
    return error;
}

cgrad_error relu_forward(const struct tensor* const x, struct tensor* const out)
{
    if (!x || !out)
    {
        return TENSOR_NULL;
    }
    if (!x->data || !out->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (!x->shape || !out->shape)
    {
        return TENSOR_SHAPE_NULL;
    }
    if (!tensor_same_shape(x, out))
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    // Avoid multiple indirections for performance
    double* x_data = x->data;
    double* out_data = out->data;
    size_t out_data_size = out->data_size;

    for (size_t i = 0; i < out_data_size; i++)
    {
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }

    return NO_ERROR;
}

static void relu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor* const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *const x = ctx->operands[RELU_ONLY_OPERAND];
    
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