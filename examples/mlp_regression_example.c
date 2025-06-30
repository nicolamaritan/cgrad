#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/mse.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "optimizers/sgd.h"
#include "utils/random.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Example dataset build
void build_example_dataset(struct tensor *x, struct tensor *y_target);
// Example 2-layer-friendly function: y = tanh(w·x + b)
double compute_example_y_target(double *x_row, double *weights, double bias, size_t dim);

int main()
{
    init_random();

    const size_t batch_size = 128;
    const size_t input_dim = 64;
    const size_t hidden_dim = 128;
    const size_t out_dim = 1;

    struct tensor *x = tensor2d_alloc(batch_size, input_dim);
    struct tensor *y_target = tensor2d_alloc(batch_size, 1);
    if (!x || !y_target) {
        tensor_free(x); 
        tensor_free(y_target);
        return 1; 
    }

    build_example_dataset(x, y_target);

    // Allocate model
    struct linear_layer *linear1 = linear_create(input_dim, hidden_dim);
    linear_xavier_init(linear1);

    struct linear_layer *linear2 = linear_create(hidden_dim, out_dim);
    linear_xavier_init(linear2);

    // Setup model params
    struct model_params params;
    init_model_params(&params);
    add_model_param(&params, linear1->weights);
    add_model_param(&params, linear1->biases);
    add_model_param(&params, linear2->weights);
    add_model_param(&params, linear2->biases);

    // Setup optimizer
    struct sgd_state opt_state;
    if (init_sgd_state(&opt_state, &params) != NO_ERROR)
        exit(1);

    double lr = 3e-4;
    double momentum = 0.9;

    size_t epochs = 100;
    for (size_t i = 0; i < epochs; i++)
    {
        // ------------- Forward -------------
        struct tensor *mult1 = tensor2d_alloc(batch_size, hidden_dim);
        struct tensor *h1 = tensor2d_alloc(batch_size, hidden_dim);
        if (linear_forward_graph(x, linear1, mult1, h1) != NO_ERROR)
            exit(1);

        struct tensor *h2 = tensor2d_alloc(batch_size, hidden_dim);
        relu_forward_graph(h1, h2); 

        struct tensor *mult3 = tensor2d_alloc(batch_size, out_dim);
        struct tensor *h3 = tensor2d_alloc(batch_size, out_dim);
        if (linear_forward_graph(h2, linear2, mult3, h3) != NO_ERROR)
            exit(1);

        struct tensor *z = tensor2d_alloc(1, 1);
        if (mse_loss_graph(h3, y_target, z) != NO_ERROR)
            exit(1);

        printf("epoch %ld, loss: %f\n", i, z->data[0]);

        // ------------- Backward -------------
        zero_grad(&params);        
        backward(z, false);
        sgd_step(lr, momentum, false, &opt_state, &params);

        // Clear iteration allocations
        tensor_free(h1);
        tensor_free(mult1);
        tensor_free(h2);
        tensor_free(h3);
        tensor_free(mult3);
        tensor_free(z);
    }

    // Cleanup
    free_sgd_state_tensors(&opt_state);
    tensor_free(x);
    tensor_free(y_target);
    linear_free(linear1);
    linear_free(linear2);
    return 0;
}

double compute_example_y_target(double *x_row, double *weights, double bias, size_t dim) 
{
    double dot = 0.0;
    for (size_t j = 0; j < dim; j++) {
        dot += x_row[j] * weights[j];
    }
    return tanh(dot + bias);
}

void build_example_dataset(struct tensor *x, struct tensor *y_target)
{
    // Random weights and bias for generating y
    double lb = -20;
    double ub = 20;
    double weights[x->shape[1]];
    for (size_t j = 0; j < x->shape[1]; j++)
    {
        weights[j] = sample_uniform(lb, ub);
    }

    double bias = sample_uniform(lb, ub);
    
    // Populate x with random values and compute y
    for (size_t i = 0; i < x->shape[0]; i++) 
    {
        double x_row[x->shape[1]];
        for (size_t j = 0; j < x->shape[1]; j++) 
        {
            double value = sample_uniform(lb, ub);
            x_row[j] = value;
            tensor2d_set_unchecked(x, i, j, value);
        }
    
        double y_value = compute_example_y_target(x_row, weights, bias, x->shape[1]);
        tensor2d_set_unchecked(y_target, i, 0, y_value);
    }
}