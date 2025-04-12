#include "autograd/backpropagation.h"
#include "autograd/computational_graph.h"
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct 
{
    computational_graph_node* targets[AUTOGRAD_MAX_TARGETS];
    size_t size;
} backpropagation_targets;

static void identify_backpropagation_nodes(computational_graph_node* const node, backpropagation_targets* targets);
static tensor* build_gradient(computational_graph_node* const node);
static void build_gradients(backpropagation_targets* const targets);
int add_target(backpropagation_targets* const targets, computational_graph_node* const node);
void set_gradient_wrt_itself(tensor* const t);

void backward(tensor* t, bool retain_graph)
{
    backpropagation_targets targets;
    targets.size = 0;

    identify_backpropagation_nodes(t->node, &targets);

    set_gradient_wrt_itself(t);
    build_gradients(&targets);

    if (retain_graph)
        return;

    for (size_t i = 0; i < targets.size; i++)
    {
        computational_graph_node* node = targets.targets[i];
        node->t->node = NULL;
        free_computational_graph_node(node);
    }
}

static void identify_backpropagation_nodes(computational_graph_node* const node, backpropagation_targets* targets)
{
    node->is_involved_in_backprop = true;
    add_target(targets, node);
    for (size_t i = 0; i < node->n_children; i++)
        identify_backpropagation_nodes(node->children[i], targets);
}

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

        computational_graph_node *parent_node = node->parents[i];
        const tensor **const operands = (const tensor** const)parent_node->tensor_operands;

        // Get which is the operand of the current node in the operation
        // that created the i-th parent. This info is stored in the current node
        size_t operand = node->parents_operands[i];
        
        // Compute gradient and add to current grad
        tensor* parent_i_gradient = tensor_no_grad_alloc(node->t->shape, node->t->shape_size);
        
        parent_node->function[operand](operands, D, parent_i_gradient);

        int terror = tensor_add_inplace(node->t->grad, parent_i_gradient);
        if (terror != NO_ERROR)
            exit(1); 

        tensor_free(parent_i_gradient);
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

int add_target(backpropagation_targets* const targets, computational_graph_node* const node)
{
    size_t const size = targets->size;
    if (size >= AUTOGRAD_MAX_TARGETS)
    {
        return 1;
    }

    targets->targets[size] = node;
    targets->size++;

    return 0;
}

void set_gradient_wrt_itself(tensor* const t)
{
    if (t->data_size == 1)
    {
        tensor2d_set_unchecked(t->grad, 0, 0, 1);
        return;
    }
    perror("Error: Not implemented yet");
    exit(1);
}