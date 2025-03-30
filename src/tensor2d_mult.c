#include "tensor.h"
#include "tensor2d_mult.h"
#include "computational_graph.h"
#include <cblas.h>
#include <stdlib.h>

void tensor2d_mult_unchecked(const tensor *const A, const tensor *const B, tensor *const out);

tensor_error tensor2d_mult(const tensor *const A, const tensor *const B, tensor *const out)
{
    if (!A || !B || !out)
        return TENSOR_NULL;
    if (!A->shape || !B->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (A->shape[1] != B->shape[0])
        return TENSOR_SHAPE_MISMATCH; // Columns of A != rows of B
    if (out->shape[0] != A->shape[0] || out->shape[1] != B->shape[1])
        return TENSOR_SHAPE_MISMATCH; // Output shape mismatch

    tensor2d_mult_unchecked(A, B, out);
    return TENSOR_OK;
}

tensor_error tensor2d_mult_graph(tensor *const A, tensor *const B, tensor *const out)
{
    tensor_error error = tensor2d_mult(A, B, out);

    if (error != TENSOR_OK)
        return error;

    // Update computational graph

    computational_graph_node *A_node = A->node ? A->node : computational_graph_node_tensor_alloc(A);
    computational_graph_node *B_node = B->node ? B->node : computational_graph_node_tensor_alloc(B);
    computational_graph_node *out_node = computational_graph_node_tensor_alloc(out);

    add_parent(A_node, out_node, LHS_TENSOR);
    add_parent(B_node, out_node, RHS_TENSOR);
    add_child(out_node, A_node);
    add_child(out_node, B_node);

    // Setup inputs
    tensor2d_mult_inputs *inputs = (tensor2d_mult_inputs *)malloc(sizeof(tensor2d_mult_inputs));
    inputs->lhs_tensor = A;
    inputs->rhs_tensor = B;

    // Setup data
    backpropagation_function_data *data = (backpropagation_function_data *)malloc(sizeof(backpropagation_function_data));
    data->layer = NULL;
    data->inputs = (void*)inputs;

    out_node->data = data;

    backpropagation_function function = (backpropagation_function)&tensor2d_mult_backpropagate;
    out_node->function = function;

    out_node->free_data = (backpropagation_function_data_cleanup)&free_tensor2d_mult_backpropagation_function_data;

    return TENSOR_OK;
}

void tensor2d_mult_unchecked(const tensor *const A, const tensor *const B, tensor *const out)
{
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A->shape[0], // M
        B->shape[1], // N
        A->shape[1], // K (must match B->shape[0])
        1.0,
        A->data,
        A->shape[1], // lda
        B->data,
        B->shape[1], // ldb
        0.0,
        out->data,
        out->shape[1] // ldc
    );
}

void tensor2d_mult_backpropagate(const backpropagation_function_data *const data, const tensor *const grad_wrt_out, tensor *grad_wrt_operand, size_t operand)
{
    tensor2d_mult_inputs *input = (tensor2d_mult_inputs *)data->inputs;

    switch (operand)
    {
    case LHS_TENSOR:
        tensor *rhs = input->rhs_tensor;
        tensor *rhs_trans= tensor2d_no_grad_alloc(rhs->shape[1], rhs->shape[0]);
        tensor2d_trans(rhs, rhs_trans);
        tensor2d_mult(grad_wrt_out, rhs_trans, grad_wrt_operand);
        tensor_no_grad_free(rhs_trans);
        break;

    case RHS_TENSOR:
        tensor* lhs = input->lhs_tensor;
        tensor *lhs_trans = tensor2d_no_grad_alloc(lhs->shape[1], lhs->shape[0]);
        tensor2d_trans(lhs, lhs_trans);
        tensor2d_mult(lhs_trans, grad_wrt_out, grad_wrt_operand);
        tensor_no_grad_free(lhs_trans);
        break;

    default:
        break;
    }
}

void free_tensor2d_mult_backpropagation_function_data(backpropagation_function_data *data)
{
    free(data->inputs);
    free(data);
}