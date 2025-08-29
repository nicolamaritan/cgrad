#include "cgrad/layers/linear.h"
#include "cgrad/tensor/tensor2d_mult.h"
#include "cgrad/tensor/tensor2d_add_row_vector.h"
#include "cgrad/tensor/tensor2d_trans.h"
#include "cgrad/tensor/tensor_sum.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include "cgrad/utils/random.h"
#include "cgrad/utils/simd_support.h"
#include <math.h>
#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>

static cgrad_error linear_xavier_init_f64(struct linear *const layer);
static cgrad_error linear_xavier_init_f32(struct linear *const layer);

cgrad_error linear_init(struct linear *const layer, const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct cgrad_env *const env)
{
    if (!layer)
    {
        return LINEAR_NULL;
    }
    if (!env)
    {
        return CGRAD_ENV_NULL;
    }

    size_t weight_shape[] = {in_dim, out_dim};
    size_t weight_shape_size = 2;
    struct tensor *weight = tensor_allocator_alloc(&env->tensor_alloc, weight_shape, weight_shape_size, dtype);
    if (!weight)
    {
        free(layer);
        return TENSOR_ALLOCATION_FAILED;
    }

    size_t bias_shape[] = {1, out_dim};
    size_t bias_shape_size = 2;
    struct tensor *bias = tensor_allocator_alloc(&env->tensor_alloc, bias_shape, bias_shape_size, dtype);
    if (!bias)
    {
        free(layer);
        tensor_allocator_free(&env->tensor_alloc, weight);
        return TENSOR_ALLOCATION_FAILED;
    }

    layer->env = env;
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->weight = weight;
    layer->bias = bias;

    return NO_ERROR;
}

cgrad_error linear_forward(struct linear *const layer, struct tensor *const x, struct tensor **const out, const bool track_grad)
{
    if (!layer)
    {
        return LINEAR_NULL;
    }
    if (!out)
    {
        return LINEAR_OUT_NULL;
    }

    // XW computation 
    struct tensor *mult = NULL;
    cgrad_error err = tensor2d_mult(x, layer->weight, &mult, track_grad, layer->env);
    if (err != NO_ERROR)
    {
        return err;
    }

    // XW + b computation
    err = tensor2d_add_row_vector(mult, layer->bias, out, track_grad, layer->env);
    if (err != NO_ERROR)
    {
        return err;
    }

    return tensor_list_add(layer->env->tensor_alloc_intermediates, mult);
}

cgrad_error linear_xavier_init(struct linear *const layer)
{
    if (!layer)
    {
        return LINEAR_NULL;
    }

    switch (layer->weight->dtype)
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
    double *restrict data = layer->weight->data;
    size_t in_dim = layer->in_dim;
    size_t out_dim = layer->out_dim;
    size_t data_size = layer->weight->data_size;

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
    float *restrict data = layer->weight->data;
    size_t in_dim = layer->in_dim;
    size_t out_dim = layer->out_dim;
    size_t data_size = layer->weight->data_size;

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

    tensor_allocator_free(&layer->env->tensor_alloc, layer->weight);
    tensor_allocator_free(&layer->env->tensor_alloc, layer->bias);
}