#include "layers/linear.h"
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

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_0
    #include "immintrin.h"
#endif

typedef enum linear_layer_operand
{
    INPUT,
    WEIGHTS,
    BIAS,
} linear_layer_operand;

static cgrad_error linear_update_computational_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const out);
static cgrad_error linear_backpropagate_input(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error linear_backpropagate_weights(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error linear_backpropagate_bias(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error linear_xavier_init_f64(struct linear_layer *layer);
static cgrad_error linear_xavier_init_f32(struct linear_layer *layer);

struct linear_layer *linear_alloc(const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct tensor_allocator *params_allocator, struct autograd_allocators *const ag_allocators)
{
    struct linear_layer *layer = (struct linear_layer *)malloc(sizeof(struct linear_layer));
    if (!layer)
    {
        return NULL;
    }

    size_t weights_shape[] = {in_dim, out_dim};
    size_t weights_shape_size = 2;
    struct tensor *weights = tensor_allocator_alloc(params_allocator, weights_shape, weights_shape_size, dtype);
    if (!weights)
    {
        free(layer);
        return NULL;
    }

    size_t biases_shape[] = {out_dim, 1};
    size_t biases_shape_size = 2;
    struct tensor *biases = tensor_allocator_alloc(params_allocator, biases_shape, biases_shape_size, dtype);
    if (!biases)
    {
        free(layer);
        tensor_allocator_free(params_allocator, weights);
        return NULL;
    }

    layer->params_allocator = params_allocator;
    layer->ag_allocators = ag_allocators;
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->weights = weights;
    layer->biases = biases;
    return layer;
}

cgrad_error linear_forward_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const out)
{
    // XW+b computation 
    cgrad_error err = linear_forward(x, layer, out);
    if (err != NO_ERROR)
    {
        return err;
    }

    return linear_update_computational_graph(x, layer, out);
}

cgrad_error linear_forward(const struct tensor *const x, const struct linear_layer *const layer, struct tensor *const out)
{
    // XW computation 
    cgrad_error error = tensor2d_mult(x, layer->weights, out);
    if (error != NO_ERROR)
    {
        return error;
    }

    // XW + b computation
    return tensor2d_add_row_vector(out, layer->biases, out);
}

cgrad_error linear_xavier_init(struct linear_layer *layer)
{
    if (!layer)
    {
        return LINEAR_LAYER_NULL;
    }

    switch (layer->weights->dtype)
    {
    case DTYPE_FLOAT64:
        return linear_xavier_init_f64(layer);
    case DTYPE_FLOAT32:
        return linear_xavier_init_f32(layer);
    default:
        return LINEAR_LAYER_INVALID_DTYPE; 
    }
}

static cgrad_error linear_xavier_init_f64(struct linear_layer *layer)
{
    double *data = layer->weights->data;
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

static cgrad_error linear_xavier_init_f32(struct linear_layer *layer)
{
    float *data = layer->weights->data;
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

void linear_free(struct linear_layer *layer)
{
    if (!layer)
    {
        return;
    }

    tensor_allocator_free(layer->params_allocator, layer->weights);
    tensor_allocator_free(layer->params_allocator, layer->biases);
    free(layer);
}

static cgrad_error linear_update_computational_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const out)
{
    cgrad_error err = add_computational_graph_link(x, INPUT, out, &linear_backpropagate_input, layer->ag_allocators);
    if (err != NO_ERROR) 
    {
        return err;
    }

    err = add_computational_graph_link(layer->weights, WEIGHTS, out, &linear_backpropagate_weights, layer->ag_allocators);
    if (err != NO_ERROR) 
    {
        return err;
    }

    return add_computational_graph_link(layer->biases, BIAS, out, &linear_backpropagate_bias, layer->ag_allocators);
}

static cgrad_error linear_backpropagate_input(const struct backpropagation_context* const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    cgrad_error err = NO_ERROR;

    const struct tensor *rhs = ctx->operands[WEIGHTS];
    if (!rhs)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_OPERAND_NULL;
    }

    struct tensor_allocator *t_allocator = ctx->owned_allocator;

    size_t shape[] = {rhs->shape[1], rhs->shape[0]};
    size_t shape_size = 2;
    struct tensor *rhs_trans = tensor_allocator_no_grad_alloc(t_allocator, shape, shape_size, grad_wrt_out->dtype);
    if (!rhs_trans)
    {
        return AUTOGRAD_BACKPROPAGATION_ALLOCATION_FAILED;
    }

    if ((err = tensor2d_trans(rhs, rhs_trans)) != NO_ERROR)
    {
        return err;
    }
    if ((err = tensor2d_mult(grad_wrt_out, rhs_trans, grad_wrt_operand)) != NO_ERROR)
    {
        return err;
    }

    tensor_allocator_no_grad_free(t_allocator, rhs_trans);

    return NO_ERROR;
}

static cgrad_error linear_backpropagate_weights(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    cgrad_error err = NO_ERROR;

    const struct tensor* lhs = ctx->operands[INPUT];
    struct tensor_allocator *t_allocator = ctx->owned_allocator;

    size_t shape[] = {lhs->shape[1], lhs->shape[0]};
    size_t shape_size = 2;
    struct tensor *lhs_trans = tensor_allocator_no_grad_alloc(t_allocator, shape, shape_size, grad_wrt_out->dtype);
    if (!lhs_trans)
    {
        return AUTOGRAD_BACKPROPAGATION_ALLOCATION_FAILED;
    }

    if ((err = tensor2d_trans(lhs, lhs_trans)) != NO_ERROR)
    {
        return err;
    }
    if ((err = tensor2d_mult(lhs_trans, grad_wrt_out, grad_wrt_operand)) != NO_ERROR)
    {
        return err;
    }

    tensor_allocator_no_grad_free(t_allocator, lhs_trans);

    return NO_ERROR;
}

static cgrad_error linear_backpropagate_bias(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    tensor_sum(grad_wrt_out, 0, grad_wrt_operand);

    return NO_ERROR;
}