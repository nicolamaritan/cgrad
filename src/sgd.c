#include "sgd.h"

//void sgd_step(double lr, backpropagation_targets* targets)
void sgd_step(double lr, model_params* params)
{
    for (size_t i = 0; i < params->size; i++)
    {
        tensor* param = params->params[i];
        tensor* gradient = param->grad;
        size_t gradient_size = gradient->data_size;

        // Ineficient, currently i am passing twice the gradient data. TODO create helper function for single pass
        double* data = gradient->data;
        for (size_t i = 0; i < gradient_size; i++)
        {
            data[i] *= -lr;
        }

        tensor_add_inplace(param, gradient);
    }
}