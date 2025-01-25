#include "linear.h"
#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>

linear_layer* linear_create(size_t in_dim, size_t out_dim)
{
    linear_layer* layer = (linear_layer*)malloc(sizeof(linear_layer));
    tensor* weights = tensor2d_alloc(in_dim, out_dim);
    tensor* biases = tensor2d_alloc(out_dim, 1);
    
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

void linear_forward_graph(const tensor* const x, const linear_layer* const layer, tensor* const out, computational_graph_node* x_node, computational_graph_node* out_node, target_computational_graph_nodes* const targets, grad_table* table)
{
    linear_forward(x, layer, out);

    // Create children node, the parameters nodes in this case
    computational_graph_node* weights_node = computational_graph_node_alloc();
    computational_graph_node* biases_node = computational_graph_node_alloc();

    grad_table_entry weights_entry;
    weights_entry.grad = NULL;
    weights_node->grad_table_index = table->n_entries;
    weights_node->t = layer->weights;
    add_entry(table, weights_entry);

    grad_table_entry biases_entry;
    biases_entry.grad = NULL;
    biases_node->grad_table_index = table->n_entries;
    biases_node->t = layer->biases;
    add_entry(table, biases_entry);

    grad_table_entry out_entry;
    out_entry.grad = NULL;
    //out_entry.grad = tensor2d_alloc(out->shape[0], out->shape[1]);
    out_node->grad_table_index = table->n_entries;
    out_node->t = out;
    add_entry(table, out_entry);    

    // Setup connections
    add_parent(x_node, out_node, PREDICTED);
    add_parent(weights_node, out_node, WEIGHTS);
    add_parent(biases_node, out_node, BIASES);
    add_child(out_node, x_node);
    add_child(out_node, weights_node);
    add_child(out_node, biases_node);

    // Setup targets
    add_target(targets, weights_node);
    add_target(targets, biases_node);

    backpropagation_function_data* data = malloc(sizeof(backpropagation_function_data));
    data->layer = (void*)layer;
    data->inputs = (void*)x;
    out_node->data = data;

    backpropagation_function function = (backpropagation_function)&linear_backpropagate;
    out_node->function = function;
}

tensor* linear_backpropagate(const backpropagation_function_data* const data, const tensor* const D, size_t operand)
{
    tensor* x = (tensor*)data->inputs;
    linear_layer* layer = (linear_layer*)data->layer;
    tensor* out;

    switch (operand)
    {
    case PREDICTED:
        tensor* weights = layer->weights;
        tensor* weights_trans = tensor2d_alloc(weights->shape[1], weights->shape[0]);
        tensor2d_trans(weights, weights_trans);
        out = tensor2d_alloc(D->shape[0], weights_trans->shape[1]);
        tensor2d_mult(D, weights_trans, out);
        tensor_free(weights_trans);
        break;

    case WEIGHTS:
        tensor* x_trans = tensor2d_alloc(x->shape[1], x->shape[0]);
        tensor2d_trans(x, x_trans);
        out = tensor2d_alloc(x_trans->shape[0], D->shape[1]);
        tensor2d_mult(x_trans, D, out);
        tensor_free(x_trans);
        break;
    
    case BIASES:
        size_t G_rows = D->shape[0];
        size_t G_cols = D->shape[1];
        out = tensor2d_alloc(1, G_cols);
        
        for (size_t j = 0; j < G_cols; j++)
            out->data[j] = 0;

        // Iterating by row since vectors are stored in row-major
        for (size_t i = 0; i < G_rows; i++)
            for (size_t j = 0; j < G_cols; j++)
                out->data[j] += D->data[i * G_cols + j];
        break;
    default:
        break;
    }

    return out;
}

void linear_forward(const tensor* const x, const linear_layer* const layer, tensor* const out)
{
    tensor2d_mult(x, layer->weights, out);
    tensor2d_add_row_vector(out, layer->biases);
}

void linear_free(linear_layer* layer) 
{
    tensor_free(layer->weights);
    tensor_free(layer->biases);
    free(layer);
}