#include "cgrad/tensor/tensor.h"
#include "cgrad/tensor/tensor2d_mult.h"
#include "cgrad/tensor/tensor2d_mult_rhs_trans.h"
#include "cgrad/tensor/tensor2d_mult_lhs_trans.h"
#include "cgrad/tensor/tensor2d_trans.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include <cblas.h>
#include <stdlib.h>

typedef enum tensor2d_mult_operand
{
    LHS_TENSOR,
    RHS_TENSOR,
} tensor2d_mult_operand;

static inline cgrad_error tensor2d_mult_update_graph(struct tensor *const x, struct tensor *const y, struct tensor **const out, struct cgrad_env *const env);
static inline cgrad_error tensor2d_mult_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor2d_mult_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor2d_mult_f32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor2d_mult_backpropagate_lhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error tensor2d_mult_backpropagate_rhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor2d_mult(struct tensor *const x, struct tensor *const y, struct tensor **const out, const bool track_grad, struct cgrad_env *const env)
{
    if (!x || !y)
    {
        return TENSOR_NULL;
    }
    if (!x->data || !y->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (x->shape[1] != y->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (x->dtype != y->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    const size_t shape[] = {x->shape[0], y->shape[1]};
    const size_t shape_size = 2;
    (*out) = tensor_allocator_alloc(&env->tensor_alloc, shape, shape_size, x->dtype);

    if (!(*out))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    cgrad_error err = tensor2d_mult_dispatch(x, y, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor2d_mult_update_graph(x, y, out, env);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor2d_mult_update_graph(struct tensor *const x, struct tensor *const y, struct tensor **const out, struct cgrad_env *const env)
{
    cgrad_error err = add_computational_graph_link(x, LHS_TENSOR, *out, &tensor2d_mult_backpropagate_lhs, env);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(y, RHS_TENSOR, *out, &tensor2d_mult_backpropagate_rhs, env);

    return err;
}

cgrad_error tensor2d_mult_into(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    if (!x || !y || !out)
    {
        return TENSOR_NULL;
    }
    if (x->shape[1] != y->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != x->shape[0] || out->shape[1] != y->shape[1])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (x->dtype != y->dtype && x->dtype != out->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor2d_mult_dispatch(x, y, out);
}

static inline cgrad_error tensor2d_mult_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    switch (x->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor2d_mult_f64(x, y, out);
    case DTYPE_FLOAT32:
        return tensor2d_mult_f32(x, y, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor2d_mult_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
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

    return NO_ERROR;
}

static cgrad_error tensor2d_mult_f32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        x->shape[0], // M
        y->shape[1], // N
        x->shape[1], // K (must match y->shape[0])
        1.0,
        (float *)x->data,
        x->shape[1], // lda
        y->data,
        y->shape[1], // ldb
        0.0,
        (float *)out->data,
        y->shape[1] // ldc
    );

    return NO_ERROR;
}

static cgrad_error tensor2d_mult_backpropagate_lhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *rhs = ctx->operands[RHS_TENSOR];
    if (!rhs)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_OPERAND_NULL;
    }
    
    /**
     * If C = A*B, then
     * dz/dA = dz/dC * B^T, hence the trans
     */
    return tensor2d_mult_rhs_trans_into(grad_wrt_out, rhs, grad_wrt_operand);
}

static cgrad_error tensor2d_mult_backpropagate_rhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *lhs = ctx->operands[LHS_TENSOR];
    if (!lhs)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_OPERAND_NULL;
    }

    /**
     * If C = A*B, then
     * dz/dB = A^T * dz/dC, hence the trans
     */
    return tensor2d_mult_lhs_trans_into(lhs, grad_wrt_out, grad_wrt_operand);
}