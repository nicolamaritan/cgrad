#include "layers/linear.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_add_row_vector.h"
#include "tensor/tensor2d_trans.h"
#include "utils/random.h"
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

static cgrad_error linear_update_computational_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const out);
static void linear_backpropagate_input(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void linear_backpropagate_weights(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void linear_backpropagate_bias(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

struct linear_layer *linear_alloc(size_t in_dim, size_t out_dim)
{
    struct linear_layer *layer = (struct linear_layer *)malloc(sizeof(struct linear_layer));
    if (!layer)
    {
        return NULL;
    }

    struct tensor *weights = tensor2d_alloc(in_dim, out_dim);
    if (!weights)
    {
        free(layer);
        return NULL;
    }

    struct tensor *biases = tensor2d_alloc(out_dim, 1);
    if (!biases)
    {
        free(layer);
        tensor_no_grad_free(weights);
        return NULL;
    }

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

void linear_xavier_init(struct linear_layer *layer)
{
    const double XAVIER_INIT_NUMERATOR = 6.0;
    double *data = layer->weights->data;
    size_t in_dim = layer->in_dim;
    size_t out_dim = layer->out_dim;
    size_t data_size = layer->weights->data_size;

    double xavier_init_bound = sqrt(XAVIER_INIT_NUMERATOR / (in_dim + out_dim));

    for (size_t i = 0; i < data_size; i++)
    {
        data[i] = sample_uniform(-xavier_init_bound, xavier_init_bound);
    }
}

void linear_free(struct linear_layer *layer)
{
    tensor_free(layer->weights);
    tensor_free(layer->biases);
    free(layer);
}

static cgrad_error linear_update_computational_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const out)
{
    cgrad_error err = add_computational_graph_link(x, INPUT, out, &linear_backpropagate_input);
    if (err != NO_ERROR) 
    {
        return err;
    }

    err = add_computational_graph_link(layer->weights, WEIGHTS, out, &linear_backpropagate_weights);
    if (err != NO_ERROR) 
    {
        return err;
    }

    return add_computational_graph_link(layer->biases, BIAS, out, &linear_backpropagate_bias);
}

static void linear_backpropagate_input(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *rhs = operands[WEIGHTS];
    struct tensor *rhs_trans= tensor2d_no_grad_alloc(rhs->shape[1], rhs->shape[0]);
    tensor2d_trans(rhs, rhs_trans);
    tensor2d_mult(grad_wrt_out, rhs_trans, grad_wrt_operand);
    tensor_no_grad_free(rhs_trans);
}

static void linear_backpropagate_weights(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor* lhs = operands[INPUT];
    struct tensor *lhs_trans = tensor2d_no_grad_alloc(lhs->shape[1], lhs->shape[0]);
    tensor2d_trans(lhs, lhs_trans);
    tensor2d_mult(lhs_trans, grad_wrt_out, grad_wrt_operand);
    tensor_no_grad_free(lhs_trans);
}

static void linear_backpropagate_bias(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    size_t G_rows = grad_wrt_out->shape[0];
    size_t G_cols = grad_wrt_out->shape[1];

    for (size_t j = 0; j < G_cols; j++)
    {
        grad_wrt_operand->data[j] = 0;
    }

    // Iterating by row since vectors are stored in row-major
    for (size_t i = 0; i < G_rows; i++)
    {
        for (size_t j = 0; j < G_cols; j++)
        {
            grad_wrt_operand->data[j] += grad_wrt_out->data[i * G_cols + j];
        }
    }
}