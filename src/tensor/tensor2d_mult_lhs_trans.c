#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_trans.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <cblas.h>
#include <stdlib.h>

static inline cgrad_error tensor2d_mult_lhs_trans_dispatch(const struct tensor *const x_trans, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor2d_mult_lhs_trans_f64(const struct tensor *const x_trans, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor2d_mult_lhs_trans_f32(const struct tensor *const x_trans, const struct tensor *const y, struct tensor *const out);

cgrad_error tensor2d_mult_lhs_trans_into(const struct tensor *const x_trans, const struct tensor *const y, struct tensor *const out)
{
    if (!x_trans || !y || !out)
    {
        return TENSOR_NULL;
    }
    if (x_trans->shape[0] != y->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (out->shape[0] != x_trans->shape[1] || out->shape[1] != y->shape[1])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (x_trans->dtype != y->dtype && x_trans->dtype != out->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor2d_mult_lhs_trans_dispatch(x_trans, y, out);
}

static inline cgrad_error tensor2d_mult_lhs_trans_dispatch(const struct tensor *const x_trans, const struct tensor *const y, struct tensor *const out)
{
    switch (x_trans->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor2d_mult_lhs_trans_f64(x_trans, y, out);
    case DTYPE_FLOAT32:
        return tensor2d_mult_lhs_trans_f32(x_trans, y, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor2d_mult_lhs_trans_f64(const struct tensor *const x_trans, const struct tensor *const y, struct tensor *const out)
{
    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        x_trans->shape[1], 
        y->shape[1],
        x_trans->shape[0], 
        1.0,
        (double *)x_trans->data,
        x_trans->shape[1], 
        y->data,
        y->shape[1], 
        0.0,
        (double *)out->data,
        out->shape[1]
    );

    return NO_ERROR;
}

static cgrad_error tensor2d_mult_lhs_trans_f32(const struct tensor *const x_trans, const struct tensor *const y, struct tensor *const out)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        x_trans->shape[1], 
        y->shape[1],
        x_trans->shape[0], 
        1.0,
        (float *)x_trans->data,
        x_trans->shape[1], 
        y->data,
        y->shape[1], 
        0.0,
        (float *)out->data,
        out->shape[1]
    );

    return NO_ERROR;
}