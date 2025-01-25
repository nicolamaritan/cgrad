#include "linear.h"
#include "mse.h"
#include "computational_graph.h"
#include "backpropagation.h"
#include "tensor.h"
#include "sgd.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    size_t in_dim = 4;
    size_t out_dim = 1;
    size_t batch_size = 2;

    // Input tensor
    printf("x: ");
    tensor* x = tensor2d_alloc(batch_size, in_dim);
    for (size_t i = 0; i < batch_size * in_dim; i++)
        x->data[i] = i+1;
    print_tensor(x);

    // Tensor
    printf("y: ");
    tensor* y_target = tensor2d_alloc(batch_size, 1);
    y_target->data[0] = 1;
    y_target->data[1] = 2;
    print_tensor(y_target);

    linear_layer* linear1 = linear_create(in_dim, out_dim);

    for (size_t i = 0; i < in_dim * out_dim; i++) 
        linear1->weights->data[i] = 1;

    printf("weights: ");
    print_tensor(linear1->weights);
    printf("biases: ");
    print_tensor(linear1->biases);

    // With graph
    grad_table table;
    init_grad_table(&table);
    grad_table_print(&table);

    target_computational_graph_nodes targets;
    targets.size = 0;

    computational_graph_node* x_node = computational_graph_node_alloc();
    x_node->grad_table_index = table.n_entries;
    x_node->t = x;

    grad_table_entry x_entry;
    add_entry(&table, x_entry);

    tensor* h1 = tensor2d_alloc(batch_size, out_dim);
    computational_graph_node* h1_node = computational_graph_node_alloc();
    linear_forward_graph(x, linear1, h1, x_node, h1_node, &targets, &table);

    tensor* z = tensor2d_alloc(1, 1);
    computational_graph_node* z_node = computational_graph_node_alloc();
    mse_loss_graph(h1, y_target, z, h1_node, z_node, &table);
    
    printf("h1: ");
    print_tensor(h1);

    printf("y_target: ");
    print_tensor(y_target);

    printf("z: ");
    print_tensor(z);

    grad_table_print(&table);

    print_computational_graph_node(x_node);
    print_computational_graph_node(h1_node);
    print_computational_graph_node(h1_node->children[1]);
    print_computational_graph_node(h1_node->children[2]);
    print_computational_graph_node(h1_node->parents[0]);
    print_computational_graph_node(z_node->children[1]);

    backpropagation(&targets, &table);
    grad_table_print(&table);

    sgd_step(0.001, &table, &targets);   
    printf("weights: ");
    print_tensor(linear1->weights);
    printf("biases: ");
    print_tensor(linear1->biases);
}