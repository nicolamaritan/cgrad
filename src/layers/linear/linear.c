#include "layers/linear/linear.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_add_row_vector.h"
#include "tensor/tensor2d_trans.h"
#include "tensor/tensor_sum.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include "utils/random.h"
#include "utils/simd_support.h"
#include <math.h>
#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>

typedef enum linear_layer_operand
{
    INPUT,
    WEIGHTS,
    BIAS,
} linear_layer_operand;

static cgrad_error linear_xavier_init_f64(struct linear *const layer);
static cgrad_error linear_xavier_init_f32(struct linear *const layer);

cgrad_error linear_init(struct linear *const layer, const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct tensor_allocator *const params_allocator, struct allocators *const allocs)
{
    if (!layer)
    {
        return LINEAR_NULL;
    }

    cgrad_error err = allocators_is_valid(allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    size_t weights_shape[] = {in_dim, out_dim};
    size_t weights_shape_size = 2;
    struct tensor *weights = tensor_allocator_alloc(params_allocator, weights_shape, weights_shape_size, dtype);
    if (!weights)
    {
        free(layer);
        return TENSOR_ALLOCATION_FAILED;
    }

    size_t biases_shape[] = {out_dim, 1};
    size_t biases_shape_size = 2;
    struct tensor *biases = tensor_allocator_alloc(params_allocator, biases_shape, biases_shape_size, dtype);
    if (!biases)
    {
        free(layer);
        tensor_allocator_free(params_allocator, weights);
        return TENSOR_ALLOCATION_FAILED;
    }

    layer->params_allocator = params_allocator;
    layer->allocs = allocs;
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->weights = weights;
    layer->biases = biases;

    return NO_ERROR;
}

cgrad_error linear_forward(struct linear *const layer, struct tensor *const x, struct linear_out *const out, const bool track_grad)
{
    if (!layer)
    {
        return LINEAR_NULL;
    }
    if (!out)
    {
        return LINEAR_OUT_NULL;
    }

    // Register layer allocator as out allocator
    out->tensor_alloc = layer->allocs->tensor_alloc;

    // XW computation 
    cgrad_error error = tensor2d_mult(x, layer->weights, &out->mult, track_grad, layer->allocs);
    if (error != NO_ERROR)
    {
        return error;
    }

    // XW + b computation
    return tensor2d_add_row_vector(out->mult, layer->biases, &out->result, track_grad, layer->allocs);
}

cgrad_error linear_xavier_init(struct linear *const layer)
{
    if (!layer)
    {
        return LINEAR_NULL;
    }

    switch (layer->weights->dtype)
    {
    case DTYPE_FLOAT64:
        return linear_xavier_init_f64(layer);
    case DTYPE_FLOAT32:
        return linear_xavier_init_f32(layer);
    default:
        return LINEAR_INVALID_DTYPE; 
    }
}

static cgrad_error linear_xavier_init_f64(struct linear *const layer)
{
    double *restrict data = layer->weights->data;
    size_t in_dim = layer->in_dim;
    size_t out_dim = layer->out_dim;
    size_t data_size = layer->weights->data_size;

    const double XAVIER_INIT_NUMERATOR = 6.0;
    double xavier_init_bound = sqrt(XAVIER_INIT_NUMERATOR / (in_dim + out_dim));

    for (size_t i = 0; i < data_size; i++)
    {
        data[i] = sample_uniform(-xavier_init_bound, xavier_init_bound);
    }

    return NO_ERROR;
}

static cgrad_error linear_xavier_init_f32(struct linear *const layer)
{
    float *restrict data = layer->weights->data;
    size_t in_dim = layer->in_dim;
    size_t out_dim = layer->out_dim;
    size_t data_size = layer->weights->data_size;

    const float XAVIER_INIT_NUMERATOR = 6.0;
    float xavier_init_bound = sqrt(XAVIER_INIT_NUMERATOR / (in_dim + out_dim));

    for (size_t i = 0; i < data_size; i++)
    {
        data[i] = sample_uniform(-xavier_init_bound, xavier_init_bound);
    }

    return NO_ERROR;
}

void linear_cleanup(struct linear *const layer)
{
    if (!layer)
    {
        return;
    }

    tensor_allocator_free(layer->params_allocator, layer->weights);
    tensor_allocator_free(layer->params_allocator, layer->biases);
}