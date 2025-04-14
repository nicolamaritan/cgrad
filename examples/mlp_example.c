#include "layers/linear.h"
#include "loss/mse.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_add_row_vector.h"
#include "optimizers/sgd.h"
#include "utils/random.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Example 2-layer-friendly function: y = tanh(w·x + b)
double compute_y(double *x_row, double *weights, double bias, size_t dim) 
{
    double dot = 0.0;
    for (size_t j = 0; j < dim; j++) {
        dot += x_row[j] * weights[j];
    }
    return tanh(dot + bias);
}
int main()
{
    init_random();

    const size_t batch_size = 128;
    const size_t input_dim = 4;

    tensor *x = tensor2d_alloc(batch_size, input_dim);
    tensor *y_target = tensor2d_alloc(batch_size, 1);
    if (!x || !y_target) {
        tensor_free(x); 
        tensor_free(y_target);
        return 1; 
    }
    
    // Random weights and bias for generating y
    double lb = -5;
    double ub = 5;
    double weights[4] = {sample_uniform(lb, ub), sample_uniform(lb, ub), sample_uniform(lb, ub), sample_uniform(lb, ub)};
    double bias = sample_uniform(lb, ub);
    
    // Populate x with random values and compute y
    for (size_t i = 0; i < batch_size; i++) {
        double x_row[4];
        for (size_t j = 0; j < 4; j++) {
            double value = sample_uniform(lb, ub);
            x_row[j] = value;
            tensor2d_set_unchecked(x, i, j, value);
        }
    
        double y_value = compute_y(x_row, weights, bias, input_dim);
        tensor2d_set_unchecked(y_target, i, 0, y_value);
    }

    printf("x:\n");
    print_tensor(x);
    printf("y_target:\n");
    print_tensor(y_target);

    size_t in_dim_1 = 4;
    size_t out_dim_1 = 8;
    linear_layer *linear1 = linear_create(in_dim_1, out_dim_1);
    linear_xavier_init(linear1);

    size_t in_dim_2 = 8;
    size_t out_dim_2 = 1;
    linear_layer *linear2 = linear_create(in_dim_2, out_dim_2);
    linear_xavier_init(linear2);

    model_params params;
    init_model_params(&params);
    add_model_param(&params, linear1->weights);
    add_model_param(&params, linear1->biases);
    add_model_param(&params, linear2->biases);
    add_model_param(&params, linear2->biases);

    sgd_state opt_state;
    if (init_sgd_state(&opt_state, &params) != NO_ERROR)
        exit(1);

    size_t epochs = 10000;
    for (size_t i = 0; i < epochs; i++)
    {
        tensor *mult1 = tensor2d_alloc(batch_size, out_dim_1);
        tensor *h1 = tensor2d_alloc(batch_size, out_dim_1);
        if (linear_forward_graph(x, linear1, mult1, h1) != NO_ERROR)
            exit(1);

        tensor *mult2 = tensor2d_alloc(batch_size, out_dim_2);
        tensor *h2 = tensor2d_alloc(batch_size, out_dim_2);
        if (linear_forward_graph(h1, linear2, mult2, h2) != NO_ERROR)
            exit(1);

        tensor *z = tensor2d_alloc(1, 1);
        if (mse_loss_graph(h2, y_target, z) != NO_ERROR)
            exit(1);

        printf("loss: %f\n", z->data[0]);

        zero_grad(&params);        
        backward(z, false);

        sgd_step(0.0003, 0.9, false, &opt_state, &params);

        tensor_free(h1);
        tensor_free(mult1);
        tensor_free(h2);
        tensor_free(mult2);
        tensor_free(z);

    }

    // Cleanup
    tensor_free(x);
    tensor_free(y_target);
    linear_free(linear1);
    return 0;
}