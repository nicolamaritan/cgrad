#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_trans.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <cblas.h>
#include <stdlib.h>

typedef enum tensor2d_mult_operand
{
    LHS_TENSOR,
    RHS_TENSOR,
} tensor2d_mult_operand;

static cgrad_error tensor2d_mult_dispatch(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);
static void tensor2d_mult_unchecked_f64(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);
static void tensor2d_mult_backpropagate_lhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void tensor2d_mult_backpropagate_rhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor2d_mult(const struct tensor *const A, const struct tensor *const B, struct tensor *const out)
{
    if (!A || !B || !out)
    {
        return TENSOR_NULL;
    }
    if (A->shape[1] != B->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH; // Columns of A != rows of B
    }
    if (out->shape[0] != A->shape[0] || out->shape[1] != B->shape[1])
    {
        return TENSOR_SHAPE_MISMATCH; // Output shape mismatch
    }
    if (A->dtype != B->dtype && A->dtype != out->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor2d_mult_dispatch(A, B, out);
}

cgrad_error tensor2d_mult_graph(struct tensor *const A, struct tensor *const B, struct tensor *const out, struct autograd_allocators *allocators)
{
    cgrad_error err = tensor2d_mult(A, B, out);

    if (err != NO_ERROR)
    {
        return err;
    }

    // Update computational graph
    err = add_computational_graph_link(A, LHS_TENSOR, out, &tensor2d_mult_backpropagate_lhs, allocators);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(B, RHS_TENSOR, out, &tensor2d_mult_backpropagate_rhs, allocators);

    return err;
}

static cgrad_error tensor2d_mult_dispatch(const struct tensor *const A, const struct tensor *const B, struct tensor *const out)
{
    switch (A->dtype)
    {
    case DTYPE_FLOAT64:
        tensor2d_mult_unchecked_f64(A, B, out);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void tensor2d_mult_unchecked_f64(const struct tensor *const A, const struct tensor *const B, struct tensor *const out)
{
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A->shape[0], // M
        B->shape[1], // N
        A->shape[1], // K (must match B->shape[0])
        1.0,
        (double *)A->data,
        A->shape[1], // lda
        B->data,
        B->shape[1], // ldb
        0.0,
        (double *)out->data,
        out->shape[1] // ldc
    );
}

static void tensor2d_mult_backpropagate_lhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *rhs = ctx->operands[RHS_TENSOR];
    struct tensor *rhs_trans = tensor2d_no_grad_alloc(rhs->shape[1], rhs->shape[0]);
    tensor2d_trans(rhs, rhs_trans);
    tensor2d_mult(grad_wrt_out, rhs_trans, grad_wrt_operand);
    tensor_no_grad_free(rhs_trans);
}

static void tensor2d_mult_backpropagate_rhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *lhs = ctx->operands[LHS_TENSOR];
    struct tensor *lhs_trans = tensor2d_no_grad_alloc(lhs->shape[1], lhs->shape[0]);
    tensor2d_trans(lhs, lhs_trans);
    tensor2d_mult(lhs_trans, grad_wrt_out, grad_wrt_operand);
    tensor_no_grad_free(lhs_trans);
}