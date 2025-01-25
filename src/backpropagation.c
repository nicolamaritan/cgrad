#include "backpropagation.h"
#include <stdio.h>

void backpropagation(target_computational_graph_nodes* const targets, grad_table* const table)
{
    size_t size = targets->size;
    for (size_t i = 0; i < size; i++)
    {
        build_grad(targets->targets[i], table);
    }
}

tensor* build_grad(const computational_graph_node* const node, grad_table* const table)
{
    if (table->entries[node->grad_table_index].grad)
    {
        return table->entries[node->grad_table_index].grad;
    }

    tensor* parents_G[node->n_parents];

    for (size_t i = 0; i < node->n_parents; i++)
    {
        tensor* D = build_grad(node->parents[i], table);
        size_t operand = node->parents_operands[i];     // Operand info is stored in current node
        backpropagation_function_data* data = node->parents[i]->data;
        parents_G[i] = node->parents[i]->function(data, D, operand);
    }

    tensor* G = tensor_clone(parents_G[0]);
    for (size_t i = 1; i < node->n_parents; i++)
    {
        tensor_add_inplace(G, parents_G[i]);
    }
    table->entries[node->grad_table_index].grad = G;
    return G;
}