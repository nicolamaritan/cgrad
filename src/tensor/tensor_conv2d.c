#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_trans.h"
#include "tensor/tensor_trans.h"
#include "tensor/tensor_reshape.h"
#include "tensor/tensor_im2row.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <cblas.h>
#include <stdlib.h>
#include <assert.h>

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

    const size_t H_out = x->shape[2] - kernel->shape[2] + 1;
    const size_t W_out = x->shape[3] - kernel->shape[3] + 1;

    size_t K = kernel->shape[0];
    size_t C = kernel->shape[1];
    size_t R = kernel->shape[2];
    size_t S = kernel->shape[3];

    cgrad_error err = NO_ERROR;

    struct tensor *x_patches = NULL;
    err = tensor_im2row((struct tensor *)x, kernel, &x_patches, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    struct tensor *reshaped_kernel = NULL;
    const size_t KERNEL_NEW_SHAPE[] = {K, C * R * S};
    err = tensor_reshape(kernel, KERNEL_NEW_SHAPE, 2, &reshaped_kernel, track_grad, allocs);

    struct tensor *kernel_trans = NULL;
    err = tensor2d_trans(reshaped_kernel, &kernel_trans, track_grad, allocs);
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

    struct tensor *out_patches_trans_reshaped = NULL;
    const size_t OUT_PATCHES_NEW_SHAPE[] = {K, x->shape[0], H_out, W_out};
    err = tensor_reshape(out_patches_trans, OUT_PATCHES_NEW_SHAPE, 4, &out_patches_trans_reshaped, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = tensor_trans(out_patches_trans_reshaped, 0, 1, out, track_grad, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    return NO_ERROR;
}