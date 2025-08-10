#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_trans.h"
#include "tensor/tensor_trans.h"
#include "tensor/tensor_im2row.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <cblas.h>
#include <stdlib.h>
#include <assert.h>

typedef enum tensor_conv2d_operand
{
    INPUT_TENSOR,
    KERNEL_TENSOR,
} tensor_conv2d_operand;

static inline cgrad_error tensor_conv2d_update_graph(struct tensor *const x, struct tensor *const kernel, struct tensor **const out, struct allocators *const allocs);
static inline cgrad_error tensor_conv2d_dispatch(const struct tensor *const x, const struct tensor *const kernel, struct tensor **const out, struct allocators *const allocs);
// static cgrad_error tensor_conv2d_f64(const struct tensor *const x, const struct tensor *const kernel, struct tensor *const out);
static cgrad_error tensor_conv2d_f32(const struct tensor *const x, const struct tensor *const kernel, struct tensor **const out, struct allocators *const allocs);
// static cgrad_error tensor_conv2d_backpropagate_lhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
// static cgrad_error tensor_conv2d_backpropagate_rhs(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor_conv2d(struct tensor *const x, struct tensor *const kernel, struct tensor **const out, const bool track_grad, struct allocators *const allocs)
{
    if (!x || !kernel)
    {
        return TENSOR_NULL;
    }
    if (!x->data || !kernel->data)
    {
        return TENSOR_DATA_NULL;
    }
    // if (x->shape[1] != kernel->shape[1])
    // {
    //     return TENSOR_SHAPE_MISMATCH;
    // }
    if (x->dtype != kernel->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    cgrad_error err = tensor_conv2d_dispatch(x, kernel, out, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor_conv2d_update_graph(x, kernel, out, allocs);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_conv2d_update_graph(struct tensor *const x, struct tensor *const kernel, struct tensor **const out, struct allocators *const allocs)
{
    // cgrad_error err = add_computational_graph_link(x, LHS_TENSOR, *out, &tensor_conv2d_backpropagate_lhs, allocs);
    // if (err != NO_ERROR)
    // {
    //     return err;
    // }

    // err = add_computational_graph_link(kernel, RHS_TENSOR, *out, &tensor_conv2d_backpropagate_rhs, allocs);

    // return err;
    return NO_ERROR;
}

// cgrad_error tensor_conv2d_into(const struct tensor *const x, const struct tensor *const kernel, struct tensor *const out)
// {
//     if (!x || !kernel || !out)
//     {
//         return TENSOR_NULL;
//     }
//     if (x->shape[1] != kernel->shape[0])
//     {
//         return TENSOR_SHAPE_MISMATCH;
//     }
//     if (out->shape[0] != x->shape[0] || out->shape[1] != kernel->shape[1])
//     {
//         return TENSOR_SHAPE_MISMATCH;
//     }
//     if (x->dtype != kernel->dtype && x->dtype != out->dtype)
//     {
//         return TENSOR_DTYPE_MISMATCH;
//     }

//     return tensor_conv2d_dispatch(x, kernel, out);
// }

static inline cgrad_error tensor_conv2d_dispatch(const struct tensor *const x, const struct tensor *const kernel, struct tensor **const out, struct allocators *const allocs)
{
    switch (x->dtype)
    {
    // case DTYPE_FLOAT64:
    // return tensor_conv2d_f64(x, kernel, out);
    case DTYPE_FLOAT32:
        return tensor_conv2d_f32(x, kernel, out, allocs);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

// static cgrad_error tensor_conv2d_f64(const struct tensor *const x, const struct tensor *const kernel, struct tensor **const out)
// {

//     return NO_ERROR;
// }

static cgrad_error tensor_conv2d_f32(const struct tensor *const x, const struct tensor *const kernel, struct tensor **const out, struct allocators *const allocs)
{
    // const size_t H_out = x->shape[2] - kernel->shape[2] + 1;
    // const size_t W_out = x->shape[3] - kernel->shape[3] + 1;

    // size_t K = kernel->shape[0];
    // size_t C = kernel->shape[1];
    // size_t R = kernel->shape[2];
    // size_t S = kernel->shape[3];

    bool track_grad = false;
    cgrad_error err = NO_ERROR;

    struct tensor *x_patches = NULL;
    err = tensor_im2row((struct tensor *)x, kernel, &x_patches, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    // View kernel with data in row major
    // I know cast bad lol. TODO be fixed asap!
    // ((struct tensor *)kernel)->shape[1] = C * R * S;
    // ((struct tensor *)kernel)->shape_size = 2;

    struct tensor *kernel_trans = NULL;
    err = tensor2d_trans((struct tensor *)kernel, &kernel_trans, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    struct tensor *out_patches = NULL;
    err = tensor2d_mult(x_patches, kernel_trans, &out_patches, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    struct tensor *out_patches_trans = NULL;
    err = tensor2d_trans(out_patches, &out_patches_trans, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    // out_patches_trans->shape[0] = K;
    // out_patches_trans->shape[1] = x->shape[0];
    // out_patches_trans->shape[2] = H_out;
    // out_patches_trans->shape[3] = W_out;
    // out_patches_trans->shape_size = 4;

    // out_patches_trans->stride[0] = x->shape[0] * H_out * W_out;
    // out_patches_trans->stride[1] = H_out * W_out;
    // out_patches_trans->stride[2] = W_out;
    // out_patches_trans->stride[3] = 1;

    err = tensor_trans(out_patches_trans, 0, 1, out, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    // Bring back kernel to original shape
    // I know cast bad lol. TODO be fixed asap!

    // ((struct tensor *)kernel)->shape[1] = C;
    // ((struct tensor *)kernel)->shape[2] = R;
    // ((struct tensor *)kernel)->shape[3] = S;
    // ((struct tensor *)kernel)->shape_size = 4;

    return NO_ERROR;
}