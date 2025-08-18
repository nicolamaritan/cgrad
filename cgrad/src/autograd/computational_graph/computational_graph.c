#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/memory/computational_graph/computational_graph_allocator.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void print_computational_graph_node(const struct computational_graph_node *node)
{
    if (!node)
    {
        printf("[NULL NODE]\n");
        return;
    }

    printf("Node: %p\n", (void *)node);
    printf("├── Parents: %zu\n", node->n_parents);
    // for (size_t i = 0; i < node->n_parents; i++)
    // {
    //     printf("│   ├── Parent %zu: %p (operand %zu)\n",
    //            i, (void *)node->parents[i], node->parents_operands[i]);
    // }
    printf("├── Children: %zu\n", node->n_children);
    for (size_t i = 0; i < node->n_children; i++)
    {
        printf("│   ├── Child %zu: %p\n", i, (void *)node->children[i]);
    }
    printf("└── Backprop Function: %p\n\n", (void *)node->function);
}

