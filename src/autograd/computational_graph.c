#include "autograd/computational_graph.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct computational_graph_node *computational_graph_node_alloc()
{
    struct computational_graph_node *node = (struct computational_graph_node *)malloc(sizeof(struct computational_graph_node));
    if (!node)
    {
        return NULL;
    }

    node->n_children = 0;
    node->n_parents = 0;
    node->t = NULL;
    node->is_involved_in_backprop = false;
    node->is_grad_computed = false;

    // Initialize arrays to prevent undefined behavior
    memset(node->parents, 0, sizeof(node->parents));
    memset(node->children, 0, sizeof(node->children));
    memset(node->parents_operands, 0, sizeof(node->parents_operands));
    memset(node->function, 0, sizeof(node->function));
    memset(node->tensor_operands, 0, sizeof(node->tensor_operands));

    return node;
}

struct computational_graph_node *computational_graph_node_tensor_alloc(struct tensor *const t)
{
    struct computational_graph_node *node = computational_graph_node_alloc();
    if (!node)
    {
        return NULL;
    }

    t->node = node;
    node->t = t;
    return node;
}

void free_computational_graph_node(struct computational_graph_node *const node)
{
    if (node->t->node)
    {
        node->t->node = NULL;
    }

    free(node);
}

cgrad_error add_child(struct computational_graph_node *const node, struct computational_graph_node *const child)
{
    size_t const n_children = node->n_children;
    if (n_children >= AUTOGRAD_MAX_CHILDREN)
    {
        return AUTOGRAD_MAX_CHILDREN_EXCEEDED;
    }

    node->children[n_children] = child;
    node->n_children++;

    return NO_ERROR;
}

cgrad_error add_parent(struct computational_graph_node *const node, struct computational_graph_node *const parent, const size_t operand)
{
    size_t const n_parents = node->n_parents;
    if (n_parents >= AUTOGRAD_MAX_PARENTS)
    {
        return AUTOGRAD_MAX_PARENTS_EXCEEDED;
    }

    node->parents[n_parents] = parent;
    node->parents_operands[n_parents] = operand;
    node->n_parents++;

    return NO_ERROR;
}

cgrad_error add_computational_graph_link(struct tensor* operand, size_t operand_id, struct tensor* result, backpropagation_function backprop_function)
{
    struct computational_graph_node *operand_node = operand->node ? operand->node : computational_graph_node_tensor_alloc(operand);
    if (!operand_node)
    {
        return AUTOGRAD_COMPUTATIONAL_GRAPH_NODE_ALLOCATION_ERROR;
    }

    struct computational_graph_node *result_node = result->node ? result->node : computational_graph_node_tensor_alloc(result);
    if (!result_node)
    {
        free_computational_graph_node(operand_node);
        return AUTOGRAD_COMPUTATIONAL_GRAPH_NODE_ALLOCATION_ERROR;
    }

    // Setup connection
    cgrad_error error = add_parent(operand_node, result_node, operand_id);
    if (error != NO_ERROR)
    {
        free_computational_graph_node(operand_node);
        free_computational_graph_node(result_node);
        return error;
    }

    error = add_child(result_node, operand_node);
    if (error != NO_ERROR)
    {
        free_computational_graph_node(operand_node);
        free_computational_graph_node(result_node);
        return error;
    }

    // Setup backpropagation function
    result_node->function[operand_id] = backprop_function; 

    // Setup operand in the tensor operands pointer
    result_node->tensor_operands[operand_id] = operand;

    return NO_ERROR;
}

void print_computational_graph_node(const struct computational_graph_node *node)
{
    if (!node)
    {
        printf("[NULL NODE]\n");
        return;
    }

    printf("Node: %p\n", (void *)node);
    printf("├── Parents: %zu\n", node->n_parents);
    for (size_t i = 0; i < node->n_parents; i++)
    {
        printf("│   ├── Parent %zu: %p (operand %zu)\n",
               i, (void *)node->parents[i], node->parents_operands[i]);
    }
    printf("├── Children: %zu\n", node->n_children);
    for (size_t i = 0; i < node->n_children; i++)
    {
        printf("│   ├── Child %zu: %p\n", i, (void *)node->children[i]);
    }
    printf("└── Backprop Function: %p\n\n", (void *)node->function);
}