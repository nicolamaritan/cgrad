#include "layers/linear.h"
#include "layers/relu.h"
#include "losses/mse.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "optimizers/sgd.h"
#include "memory/tensor_cpu_allocator.h"
#include "memory/computational_graph_cpu_allocator.h"
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
    const int SEED = 42;
    init_random_seed(SEED);

    const size_t batch_size = 128;
    const size_t input_dim = 64;
    const size_t hidden_dim = 128;
    const size_t out_dim = 1;

    // Memory initialization
    struct tensor_cpu_pool t_pool;
    if (tensor_cpu_pool_init(&t_pool) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    struct computational_graph_cpu_pool cg_pool;
    if (computational_graph_cpu_pool_init(&cg_pool) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    // Allocator initialization
    struct tensor_allocator t_allocator = make_tensor_cpu_allocator(&t_pool);
    struct computational_graph_allocator cg_allocator = make_computational_graph_cpu_allocator(&cg_pool);

    // size_t shape[] = {2, 2};
    // struct tensor *t = tensor_allocator_alloc(&t_allocator, shape, 2);
    // struct computational_graph_node *node;
    // printf("%ld\n", sizeof(struct computational_graph_node));
    
    // for (size_t i = 0; i < 100; i++)
    // {
    //     struct computational_graph_node* next_node = computational_graph_allocator_alloc(&cg_allocator, t);
    //     printf("%td\n", (void*)next_node - (void*)node);
    //     node = next_node;
    //     // computational_graph_allocator_free(&cg_allocator, node);
    // }

    size_t x_shape[] = {batch_size, input_dim};
    size_t x_shape_size = 2;
    struct tensor *x = tensor_allocator_alloc(&t_allocator, x_shape, x_shape_size);

    size_t y_shape[] = {batch_size, 1};
    size_t y_shape_size = 2;
    struct tensor *y_target = tensor_allocator_alloc(&t_allocator, y_shape, y_shape_size);
    if (!x || !y_target)
    {
        tensor_allocator_free(&t_allocator, x);
        tensor_allocator_free(&t_allocator, y_target);
        return EXIT_FAILURE;
    }

    build_example_dataset(x, y_target);

    // Allocate model
    struct linear_layer *linear1 = linear_alloc(input_dim, hidden_dim, &t_allocator);
    linear_xavier_init(linear1);

    struct linear_layer *linear2 = linear_alloc(hidden_dim, out_dim, &t_allocator);
    linear_xavier_init(linear2);

    // Setup model params
    struct model_params params;
    init_model_params(&params);
    add_model_param(&params, linear1->weights);
    add_model_param(&params, linear1->biases);
    add_model_param(&params, linear2->weights);
    add_model_param(&params, linear2->biases);

    // Setup optimizer
    struct sgd_optimizer opt;
    if (sgd_optimizer_init(&opt, &params, &t_allocator) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    double lr = 3e-4;
    double momentum = 0.9;

    size_t epochs = 100;
    for (size_t i = 0; i < epochs; i++)
    {
        // ------------- Forward -------------
        size_t h1_shape[] = {batch_size, hidden_dim};
        size_t h1_shape_size = 2;
        struct tensor *h1 = tensor_allocator_alloc(&t_allocator, h1_shape, h1_shape_size);
        if (linear_forward_graph(x, linear1, h1) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        size_t h2_shape[] = {batch_size, hidden_dim};
        size_t h2_shape_size = 2;
        struct tensor *h2 = tensor_allocator_alloc(&t_allocator, h2_shape, h2_shape_size);
        relu_forward_graph(h1, h2);

        size_t h3_shape[] = {batch_size, out_dim};
        size_t h3_shape_size = 2;
        struct tensor *h3 = tensor_allocator_alloc(&t_allocator, h3_shape, h3_shape_size);
        if (linear_forward_graph(h2, linear2, h3) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        size_t z_shape[] = {1, 1};
        size_t z_shape_size = 2;
        struct tensor *z = tensor_allocator_alloc(&t_allocator, z_shape, z_shape_size);
        if (mse_loss_graph(h3, y_target, z) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        printf("epoch %ld, loss: %f\n", i, z->data[0]);

        // ------------- Backward -------------
        zero_grad(&params);
        backward(z, false);
        sgd_optimizer_step(&opt, lr, momentum, false);

        // Clear iteration allocations
        tensor_allocator_free(&t_allocator, h1);
        tensor_allocator_free(&t_allocator, h2);
        tensor_allocator_free(&t_allocator, h3);
        tensor_allocator_free(&t_allocator, z);
    }

    // Cleanup
    sgd_optimizer_cleanup(&opt);
    tensor_allocator_free(&t_allocator, x);
    tensor_allocator_free(&t_allocator, y_target);
    linear_free(linear1);
    linear_free(linear2);
    return EXIT_SUCCESS;
}

double compute_example_y_target(double *x_row, double *weights, double bias, size_t dim)
{
    double dot = 0.0;
    for (size_t j = 0; j < dim; j++)
    {
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