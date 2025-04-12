#include "backpropagation.h"
#include "computational_graph.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_TARGETS 1024

typedef struct 
{
    computational_graph_node* targets[MAX_TARGETS];
    size_t size;
} backpropagation_targets;

//static tensor* build_grad_old(const computational_graph_node* const node);
//static void zero_grad_node(computational_graph_node* const root);
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
    computational_graph_node* root = node;
    while (root->n_parents != 0)
        root = root->parents[0];
    printf("0. root->t->grad");
    print_tensor(root->t->grad);

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
        printf("1. root->t->grad");
        print_tensor(root->t->grad);

        tensor* D = build_gradient(node->parents[i]);

        computational_graph_node *parent_node = node->parents[i];
        const tensor **const operands = (const tensor** const)parent_node->tensor_operands;

        // Get which is the operand of the current node in the operation
        // that created the i-th parent. This info is stored in the current node
        size_t operand = node->parents_operands[i];
        
        // Compute gradient and add to current grad
        tensor* parent_i_gradient = tensor_no_grad_alloc(node->t->shape, node->t->shape_size);
        
        printf("2. root->t->grad");
        print_tensor(root->t->grad);

        parent_node->function[operand](operands, D, parent_i_gradient);

        printf("3. root->t->grad");
        print_tensor(root->t->grad);


        int terror = tensor_add_inplace(node->t->grad, parent_i_gradient);
        if (terror != TENSOR_OK)
            exit(1); 

        printf("4. root->t->grad");
        print_tensor(root->t->grad);

        tensor_free(parent_i_gradient);
    }

    printf("5. root->t->grad");
    print_tensor(root->t->grad);

    printf("%p\n%p\n%p\n", node, &node->is_grad_computed, root->t->grad->shape);

    printf("sizes: %ld, %ld, %ld\n", sizeof(node->is_grad_computed), sizeof(bool), sizeof(true));
    node->is_grad_computed = true;

    // printf("node->t->grad, node %p", node);
    // print_tensor(node->t->grad);

    printf("6. root->t->grad");
    print_tensor(root->t->grad);
    printf("data_shape: %ld, %ld, %ld\n", root->t->grad->shape[0], root->t->grad->shape[1], root->t->grad->shape[2]);

    return node->t->grad;
}

static void build_gradients(backpropagation_targets* const targets)
{
    size_t size = targets->size;
    for (size_t i = 0; i < size; i++)
    {
        printf(">>>%ld\n", i);
        build_gradient(targets->targets[i]);
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