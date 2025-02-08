#include "backpropagation.h"
#include "computational_graph.h"
#include <stdio.h>
#include <string.h>

#define MAX_TARGETS 1024

typedef struct 
{
    computational_graph_node* targets[MAX_TARGETS];
    size_t size;
} backpropagation_targets;

//static tensor* build_grad_old(const computational_graph_node* const node);
static void zero_grad_node(computational_graph_node* const root);
static void identify_backpropagation_nodes(computational_graph_node* const node, backpropagation_targets* targets);
static tensor* build_gradient(computational_graph_node* const node);
static void build_gradients(backpropagation_targets* const targets);
int add_target(backpropagation_targets* const targets, computational_graph_node* const node);

void backward(tensor* t)
{
    backpropagation_targets targets;
    targets.size = 0;

    identify_backpropagation_nodes(t->node, &targets);
    build_gradients(&targets);
}

// void backpropagate(backpropagation_targets* const targets)
// {
//     size_t size = targets->size;
//     for (size_t i = 0; i < size; i++)
//     {
//         build_grad_old(targets->targets[i]->node);
//     }
// }

static void identify_backpropagation_nodes(computational_graph_node* const node, backpropagation_targets* targets)
{
    node->is_involved_in_backprop = true;
    add_target(targets, node);
    for (size_t i = 0; i < node->n_children; i++)
        identify_backpropagation_nodes(node->children[i], targets);
}

// tensor* build_grad_old(const computational_graph_node* const node)
// {
//     if (node->t->grad)
//     {
//         // return table->entries[node->grad_table_index].grad;
//         return node->t->grad;
//     }
//     // print_computational_graph_node(node);

//     tensor* parents_G[node->n_parents];

//     for (size_t i = 0; i < node->n_parents; i++)
//     {
//         tensor* D = build_grad_old(node->parents[i]);
//         size_t operand = node->parents_operands[i];     // Operand info is stored in current node
//         backpropagation_function_data* data = node->parents[i]->data;
//         parents_G[i] = node->parents[i]->function(data, D, operand);
//     }

//     tensor* G = tensor_clone(parents_G[0]);
//     for (size_t i = 1; i < node->n_parents; i++)
//     {
//         tensor_add_inplace(G, parents_G[i]);
//     }
//     //table->entries[node->grad_table_index].grad = G;
//     node->t->grad = G;
//     return G;
// }

static tensor* build_gradient(computational_graph_node* const node)
{
    if (node->is_grad_computed)
    {
        return node->t->grad;
    }

    for (size_t i = 0; i < node->n_parents; i++)
    {
        if (!node->parents[i]->is_involved_in_backprop)
        {
            continue;
        }
        tensor* D = build_gradient(node->parents[i]);
        size_t operand = node->parents_operands[i];     // Operand info is stored in current node
        backpropagation_function_data* data = node->parents[i]->data;
        
        // Compute gradient and add to current grad
        tensor* parent_i_gradient = node->parents[i]->function(data, D, operand);
        // printf("node->t->grad: %p\n", node->t->grad);
        // printf("parent_i_gradient: %p\n", parent_i_gradient);
        tensor_add_inplace(node->t->grad, parent_i_gradient);
    }
   
    node->is_grad_computed = true;
    return node->t->grad;
}

static void build_gradients(backpropagation_targets* const targets)
{
    size_t size = targets->size;
    for (size_t i = 0; i < size; i++)
    {
        build_gradient(targets->targets[i]);
    }
}

void zero_grad(tensor* const root)
{
    zero_grad_node(root->node);
}

void zero_grad_node(computational_graph_node* const root)
{
    // print_computational_graph_node(root);
    //print_tensor(root->t->grad);

    if (!root->t->grad)
        return;

    // printf("freeing...\n");
    //tensor_free(root->t->grad);
    // printf("freed\n\n\n");
    // root->t->grad = NULL;
    memset(root->t->grad->data, 0, sizeof(double) * root->t->grad->data_size);

    size_t n_children = root->n_children;
    for (size_t i = 0; i < n_children; i++)
    {
        zero_grad_node(root->children[i]);
    }
}

int add_target(backpropagation_targets* const targets, computational_graph_node* const node)
{
    size_t const size = targets->size;
    if (size >= MAX_TARGETS)
    {
        return 1;
    }

    targets->targets[size] = node;
    targets->size++;

    return 0;
}
