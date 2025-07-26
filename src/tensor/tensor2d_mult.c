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

static cgrad_error tensor2d_mult_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static void tensor2d_mult_unchecked_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static void tensor2d_mult_backpropagate_lhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void tensor2d_mult_backpropagate_rhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor2d_mult(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    if (!x || !y || !out)
    {
        return TENSOR_NULL;
    }
    if (x->shape[1] != y->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH; // Columns of x != rows of y
    }
    if (out->shape[0] != x->shape[0] || out->shape[1] != y->shape[1])
    {
        return TENSOR_SHAPE_MISMATCH; // Output shape mismatch
    }
    if (x->dtype != y->dtype && x->dtype != out->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor2d_mult_dispatch(x, y, out);
}

cgrad_error tensor2d_mult_graph(struct tensor *const x, struct tensor *const y, struct tensor *const out, struct autograd_allocators *allocators)
{
    cgrad_error err = tensor2d_mult(x, y, out);

    if (err != NO_ERROR)
    {
        return err;
    }

    // Update computational graph
    err = add_computational_graph_link(x, LHS_TENSOR, out, &tensor2d_mult_backpropagate_lhs, allocators);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(y, RHS_TENSOR, out, &tensor2d_mult_backpropagate_rhs, allocators);

    return err;
}

static cgrad_error tensor2d_mult_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    switch (x->dtype)
    {
    case DTYPE_FLOAT64:
        tensor2d_mult_unchecked_f64(x, y, out);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void tensor2d_mult_unchecked_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        x->shape[0], // M
        y->shape[1], // N
        x->shape[1], // K (must match y->shape[0])
        1.0,
        (double *)x->data,
        x->shape[1], // lda
        y->data,
        y->shape[1], // ldb
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