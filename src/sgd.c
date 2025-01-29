#include "sgd.h"

void sgd_step(double lr, backpropagation_targets* targets)
{
    for (size_t i = 0; i < targets->size; i++)
    {
        tensor* target = targets->targets[i];
        tensor* gradient = target->grad;

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

        tensor_add_inplace(target, gradient);
    }
}