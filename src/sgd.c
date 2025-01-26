#include "sgd.h"

void sgd_step(double lr, grad_table* table, target_computational_graph_nodes* targets)
{
    for (size_t i = 0; i < targets->size; i++)
    {
        computational_graph_node* target = targets->targets[i];
        // tensor* gradient = table->entries[target->grad_table_index].grad;
        tensor* gradient = target->t->grad;

        // Compute number of elements in tensor
        size_t gradient_size = 1;
        for (size_t i = 0; gradient->shape[i]; i++)
        {
            gradient_size *= gradient->shape[i];
        }

        double* data = gradient->data;
        for (size_t i = 0; i < gradient_size; i++)
        {
            data[i] *= -lr;
        }

        tensor_add_inplace(target->t, gradient);
    }
}