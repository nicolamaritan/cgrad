#include "layers/conv2d.h"
#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_trans.h"
#include "tensor/tensor_trans.h"
#include "tensor/tensor_reshape.h"
#include "tensor/tensor_im2row.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include "utils/random.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>

static cgrad_error conv2d_xavier_init_f64(struct conv2d *const layer);
static cgrad_error conv2d_xavier_init_f32(struct conv2d *const layer);

cgrad_error conv2d_init(struct conv2d *const layer, const size_t in_channels, const size_t out_channels, const size_t kernel_size, const cgrad_dtype dtype, struct tensor_allocator *const params_allocator, struct allocators *const allocs)
{
    size_t shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    struct tensor *weight = tensor_allocator_alloc(allocs->tensor_alloc, shape, 4, dtype);

    layer->weight = weight;
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->allocs = allocs;
    layer->params_allocator = allocs->tensor_alloc;

    return NO_ERROR;
}

cgrad_error conv2d_forward(struct conv2d *const layer, struct tensor *const x, struct tensor **const out, struct tensor_list *const intermediates, const bool track_grad)
{
    struct tensor *kernel = layer->weight;

    const size_t H_out = x->shape[2] - kernel->shape[2] + 1;
    const size_t W_out = x->shape[3] - kernel->shape[3] + 1;

    size_t K = kernel->shape[0];
    size_t C = kernel->shape[1];
    size_t R = kernel->shape[2];
    size_t S = kernel->shape[3];

    cgrad_error err = NO_ERROR;

    struct tensor *x_patches = NULL;
    err = tensor_im2row((struct tensor *)x, kernel, &x_patches, track_grad, layer->allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    const size_t KERNEL_NEW_SHAPE[] = {K, C * R * S};
    struct tensor *reshaped_kernel = NULL;
    err = tensor_reshape(kernel, KERNEL_NEW_SHAPE, 2, &reshaped_kernel, track_grad, layer->allocs);

    struct tensor *kernel_trans = NULL;
    err = tensor2d_trans(reshaped_kernel, &kernel_trans, track_grad, layer->allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    struct tensor *out_patches = NULL;
    err = tensor2d_mult(x_patches, kernel_trans, &out_patches, track_grad, layer->allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    struct tensor *out_patches_trans = NULL;
    err = tensor2d_trans(out_patches, &out_patches_trans, track_grad, layer->allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    struct tensor *out_patches_trans_reshaped = NULL;
    const size_t OUT_PATCHES_NEW_SHAPE[] = {K, x->shape[0], H_out, W_out};
    err = tensor_reshape(out_patches_trans, OUT_PATCHES_NEW_SHAPE, 4, &out_patches_trans_reshaped, track_grad, layer->allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = tensor_trans(out_patches_trans_reshaped, 0, 1, out, track_grad, layer->allocs);
    if (err != NO_ERROR)
    {
        return err;
    }
    tensor_list_add(intermediates, x_patches);
    tensor_list_add(intermediates, reshaped_kernel);
    tensor_list_add(intermediates, kernel_trans);
    tensor_list_add(intermediates, out_patches);
    tensor_list_add(intermediates, out_patches_trans);
    tensor_list_add(intermediates, out_patches_trans_reshaped);

    return NO_ERROR;
}

cgrad_error conv2d_xavier_init(struct conv2d *const layer)
{
    if (!layer)
    {
        // return LINEAR_NULL;
        return 1;
    }

    switch (layer->weight->dtype)
    {
    case DTYPE_FLOAT64:
        return conv2d_xavier_init_f64(layer);
    case DTYPE_FLOAT32:
        return conv2d_xavier_init_f32(layer);
    default:
        return LINEAR_INVALID_DTYPE; 
    }
}

static cgrad_error conv2d_xavier_init_f64(struct conv2d *const layer)
{
    double *restrict data = layer->weight->data;
    size_t data_size = layer->weight->data_size;

    double xavier_init_bound = sqrt(1.0 / (layer->weight->shape[1] * layer->weight->shape[2] * layer->weight->shape[3]));

    for (size_t i = 0; i < data_size; i++)
    {
        data[i] = sample_uniform(-xavier_init_bound, xavier_init_bound);
    }

    return NO_ERROR;
}

static cgrad_error conv2d_xavier_init_f32(struct conv2d *const layer)
{
    float *restrict data = layer->weight->data;
    size_t data_size = layer->weight->data_size;

    float xavier_init_bound = sqrt(1.0 / (layer->weight->shape[1] * layer->weight->shape[2] * layer->weight->shape[3]));

    for (size_t i = 0; i < data_size; i++)
    {
        data[i] = sample_uniform(-xavier_init_bound, xavier_init_bound);
    }

    return NO_ERROR;
}

void conv2d_cleanup(struct conv2d *const layer)
{
    if (!layer)
    {
        return;
    }

    // tensor_allocator_free(layer->params_allocator, layer->weights);
    // tensor_allocator_free(layer->params_allocator, layer->biases);
}