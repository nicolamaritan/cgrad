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
    // tensor* mult = tensor2d_alloc(x->shape[0], layer->weights->shape[1]);
    tensor2d_mult_graph(x, layer->weights, out);
    return;

    // XW + b computation
    tensor2d_add_row_vector_graph(mult, layer->biases, out);
    // linear_forward(x, layer, out);

    // computational_graph_node *x_node = x->node ? x->node : computational_graph_node_tensor_alloc(x);

    // tensor *weights = layer->weights;
    // tensor *biases = layer->biases;

    // // Create children node, the parameters nodes in this case
    // computational_graph_node *weights_node = weights->node ? weights->node : computational_graph_node_tensor_alloc(weights);
    // computational_graph_node *biases_node = biases->node ? biases->node : computational_graph_node_tensor_alloc(biases);

    // computational_graph_node *out_node = computational_graph_node_tensor_alloc(out);

    // // Setup connections
    // add_parent(x_node, out_node, PREDICTED);
    // add_parent(weights_node, out_node, WEIGHTS);
    // add_parent(biases_node, out_node, BIASES);
    // add_child(out_node, x_node);
    // add_child(out_node, weights_node);
    // add_child(out_node, biases_node);

    // backpropagation_function_data *data = malloc(sizeof(backpropagation_function_data));
    // data->layer = (void *)layer;
    // data->inputs = (void *)x;
    // out_node->data = data;

    // backpropagation_function function = (backpropagation_function)&linear_backpropagate;
    // out_node->function = function;

    // out_node->free_data = (backpropagation_function_data_cleanup)&free_linear_backpropagation_function_data;
}

void linear_backpropagate(const backpropagation_function_data *const data, const tensor *const grad_wrt_out, tensor* grad_wrt_operand, size_t operand)
{
    tensor *x = (tensor *)data->inputs;
    linear_layer *layer = (linear_layer *)data->layer;
    // tensor *out;

    switch (operand)
    {
    case PREDICTED:
        tensor *weights = layer->weights;
        tensor *weights_trans = tensor2d_no_grad_alloc(weights->shape[1], weights->shape[0]);
        tensor2d_trans(weights, weights_trans);
        tensor2d_mult(grad_wrt_out, weights_trans, grad_wrt_operand);
        tensor_no_grad_free(weights_trans);
        break;

    case WEIGHTS:
        tensor *x_trans = tensor2d_no_grad_alloc(x->shape[1], x->shape[0]);
        tensor2d_trans(x, x_trans);
        tensor2d_mult(x_trans, grad_wrt_out, grad_wrt_operand);
        tensor_no_grad_free(x_trans);
        break;

    case BIASES:
        size_t G_rows = grad_wrt_out->shape[0];
        size_t G_cols = grad_wrt_out->shape[1];

        for (size_t j = 0; j < G_cols; j++)
            grad_wrt_operand->data[j] = 0;

        // Iterating by row since vectors are stored in row-major
        for (size_t i = 0; i < G_rows; i++)
            for (size_t j = 0; j < G_cols; j++)
                grad_wrt_operand->data[j] += grad_wrt_out->data[i * G_cols + j];
        break;
    default:
        break;
    }
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

void free_linear_backpropagation_function_data(backpropagation_function_data *data)
{
    free(data);
}