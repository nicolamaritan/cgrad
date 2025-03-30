#include "linear.h"
#include "random.h"
#include "tensor2d_mult.h"
#include "tensor2d_add_row_vector.h"
#include <math.h>
#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>

linear_layer *linear_create(size_t in_dim, size_t out_dim)
{
    linear_layer *layer = (linear_layer *)malloc(sizeof(linear_layer));
    tensor *weights = tensor2d_alloc(in_dim, out_dim);
    tensor *biases = tensor2d_alloc(out_dim, 1);

    if (!layer || !weights)
    {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return NULL;
    }
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->weights = weights;
    layer->biases = biases;
    return layer;
}

void linear_forward_graph(tensor *const x, linear_layer *const layer, tensor *const mult, tensor *const out)
{
    // XW computation 
    tensor2d_mult_graph(x, layer->weights, mult);

    // XW + b computation
    tensor2d_add_row_vector_graph(mult, layer->biases, out);
}

void linear_forward(const tensor *const x, const linear_layer *const layer, tensor *const mult, tensor *const out)
{
    // XW computation 
    // tensor* mult = tensor2d_alloc(x->shape[0], layer->weights->shape[1]);
    tensor2d_mult(x, layer->weights, mult);

    // XW + b computation
    tensor2d_add_row_vector(mult, layer->biases, out);
}

void linear_xavier_init(linear_layer *layer)
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

void linear_free(linear_layer *layer)
{
    tensor_free(layer->weights);
    tensor_free(layer->biases);
    free(layer);
}