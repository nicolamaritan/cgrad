#include "layers/linear.h"
#include "utils/random.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_add_row_vector.h"
#include <math.h>
#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>

const char* LINEAR_LAYER_MEMORY_ALLOCATION_ERROR = "Error: Linear layer memory allocation failed.";

struct linear_layer *linear_create(size_t in_dim, size_t out_dim)
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
        return NULL;
    }

    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->weights = weights;
    layer->biases = biases;
    return layer;
}

cgrad_error linear_forward_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const mult, struct tensor *const out)
{
    // XW computation 
    cgrad_error error = tensor2d_mult_graph(x, layer->weights, mult);
    if (error != NO_ERROR)
        return error;

    // XW + b computation
    return tensor2d_add_row_vector_graph(mult, layer->biases, out);
}

cgrad_error linear_forward(const struct tensor *const x, const struct linear_layer *const layer, struct tensor *const mult, struct tensor *const out)
{
    // XW computation 
    cgrad_error error = tensor2d_mult(x, layer->weights, mult);
    if (error != NO_ERROR)
        return error;

    // XW + b computation
    return tensor2d_add_row_vector(mult, layer->biases, out);
}

void linear_xavier_init(struct linear_layer *layer)
{
    double *data = layer->weights->data;
    size_t in_dim = layer->in_dim;
    size_t out_dim = layer->out_dim;
    size_t data_size = layer->weights->data_size;

    double xavier_init_bound = sqrt(6.0 / (in_dim + out_dim));

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