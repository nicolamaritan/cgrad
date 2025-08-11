#ifndef CONV2D_OUT_H
#define CONV2D_OUT_H

#include "memory/tensor/tensor_allocator.h"
#include "error.h"

struct conv2d_out
{
    struct tensor *x_patches;
    struct tensor *reshaped_kernel;
    struct tensor *kernel_trans;
    struct tensor *out_patches;
    struct tensor *out_patches_trans;
    struct tensor *out_patches_trans_reshaped;
    struct tensor *result;
    struct tensor_allocator *tensor_alloc;
};

#define CONV2D_OUT_INIT {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}

static inline cgrad_error conv2d_layer_out_cleanup(struct conv2d_out *const out);

static inline cgrad_error conv2d_layer_out_cleanup(struct conv2d_out *const out)
{
    if (!out)
    {
        // return LINEAR_OUT_NULL;
        return 1;
    }
    if (!out->tensor_alloc)
    {
        return TENSOR_ALLOCATOR_NULL;
    }
    tensor_allocator_free(out->tensor_alloc, out->x_patches);
    tensor_allocator_free(out->tensor_alloc, out->reshaped_kernel);
    tensor_allocator_free(out->tensor_alloc, out->kernel_trans);
    tensor_allocator_free(out->tensor_alloc, out->out_patches);
    tensor_allocator_free(out->tensor_alloc, out->out_patches_trans);
    tensor_allocator_free(out->tensor_alloc, out->out_patches_trans_reshaped);
    tensor_allocator_free(out->tensor_alloc, out->result);

    out->x_patches = NULL;
    out->reshaped_kernel = NULL;
    out->kernel_trans = NULL;
    out->out_patches = NULL;
    out->out_patches_trans = NULL;
    out->out_patches_trans_reshaped = NULL;
    out->result = NULL;

    return NO_ERROR;
}

#endif