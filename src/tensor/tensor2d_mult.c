#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_trans.h"
#include "autograd/computational_graph.h"
#include <cblas.h>
#include <stdlib.h>

void tensor2d_mult_unchecked(const struct tensor *const A, const struct tensor *const B, struct tensor *const out);

cgrad_error tensor2d_mult(const struct tensor *const A, const struct tensor *const B, struct tensor *const out)
{
    if (!A || !B || !out)
    {
        return TENSOR_NULL;
    }
    if (!A->shape || !B->shape || !out->shape)
    {
        return TENSOR_SHAPE_NULL;
    }
    if (A->shape[1] != B->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH; // Columns of A != rows of B
    }
    if (out->shape[0] != A->shape[0] || out->shape[1] != B->shape[1])
    {
        return TENSOR_SHAPE_MISMATCH; // Output shape mismatch
    }

    tensor2d_mult_unchecked(A, B, out);
    return NO_ERROR;
}

cgrad_error tensor2d_mult_graph(struct tensor *const A, struct tensor *const B, struct tensor *const out)
{
    cgrad_error err = tensor2d_mult(A, B, out);

    if (err != NO_ERROR)
    {
        return err;
    }

    // Update computational graph
    err = add_computational_graph_link(A, LHS_TENSOR, out, &tensor2d_mult_backpropagate_lhs);
    if (err != NO_ERROR)
    {
        return err;
    }
    
    err = add_computational_graph_link(B, RHS_TENSOR, out, &tensor2d_mult_backpropagate_rhs);

    return err;
}

void tensor2d_mult_unchecked(const struct tensor *const A, const struct tensor *const B, struct tensor *const out)
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

void tensor2d_mult_backpropagate_lhs(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{ 
    const struct tensor *rhs = operands[RHS_TENSOR];
    struct tensor *rhs_trans= tensor2d_no_grad_alloc(rhs->shape[1], rhs->shape[0]);
    tensor2d_trans(rhs, rhs_trans);
    tensor2d_mult(grad_wrt_out, rhs_trans, grad_wrt_operand);
    tensor_no_grad_free(rhs_trans);
}

void tensor2d_mult_backpropagate_rhs(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{ 
    const struct tensor* lhs = operands[LHS_TENSOR];
    struct tensor *lhs_trans = tensor2d_no_grad_alloc(lhs->shape[1], lhs->shape[0]);
    tensor2d_trans(lhs, lhs_trans);
    tensor2d_mult(lhs_trans, grad_wrt_out, grad_wrt_operand);
    tensor_no_grad_free(lhs_trans);
}