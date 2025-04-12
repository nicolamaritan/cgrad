#include "computational_graph.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

computational_graph_node *computational_graph_node_alloc()
{
    computational_graph_node *node = (computational_graph_node *)malloc(sizeof(computational_graph_node));
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

computational_graph_node *computational_graph_node_tensor_alloc(tensor *const t)
{
    computational_graph_node *node = computational_graph_node_alloc();
    t->node = node;
    node->t = t;
    return node;
}

void free_computational_graph_node(computational_graph_node *const node)
{
    if (node->t->node)
        node->t->node = NULL;

    // free(node->data);
    free(node);
}

int add_child(computational_graph_node *const node, computational_graph_node *const child)
{
    size_t const n_children = node->n_children;
    if (n_children >= MAX_CHILDREN)
    {
        return 1;
    }

    node->children[n_children] = child;
    node->n_children++;

    return 0;
}

int add_parent(computational_graph_node *const node, computational_graph_node *const parent, const size_t operand)
{
    size_t const n_parents = node->n_parents;
    if (n_parents >= MAX_PARENTS)
    {
        return 1;
    }

    node->parents[n_parents] = parent;
    node->parents_operands[n_parents] = operand;
    node->n_parents++;

    return 0;
}

void print_computational_graph_node(const computational_graph_node *node)
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
    // printf("├── Backprop Data: %p\n", (void *)node->data);
    printf("└── Backprop Function: %p\n\n", (void *)node->function);
}