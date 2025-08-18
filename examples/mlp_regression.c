#include "cgrad/layers/linear.h"
#include "cgrad/layers/relu.h"
#include "cgrad/losses/mse.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/memory/allocators.h"
#include "cgrad/model/model_params.h"
#include "cgrad/tensor/tensor.h"
#include "cgrad/tensor/tensor_get.h"
#include "cgrad/tensor/tensor_set.h"
#include "cgrad/optimizers/sgd.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_allocator.h"
#include "cgrad/utils/random.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Example dataset build
void build_example_dataset(struct tensor *x, struct tensor *y_target);
// Example 2-layer-friendly function: y = tanh(w·x + b)
float compute_example_y_target(float *x_row, float *weight, float bias, size_t dim);

int main()
{
    const int SEED = 42;
    init_random_seed(SEED);

    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    const size_t BATCH_SIZE = 128;
    const size_t INPUT_DIM = 64;
    const size_t HIDDEN_DIM = 128;
    const size_t OUT_DIM = 1;

    // Allocator initialization
    struct tensor_allocator tensor_alloc;
    tensor_cpu_allocator_init(&tensor_alloc);

    struct computational_graph_allocator graph_alloc;
    computational_graph_cpu_allocator_init(&graph_alloc);

    struct allocators allocs = {&tensor_alloc, &graph_alloc};

    const size_t INTERMEDIATES_CAPACITY = 20;
    struct tensor_list *intermediates = tensor_list_alloc(INTERMEDIATES_CAPACITY);

    size_t x_shape[] = {BATCH_SIZE, INPUT_DIM};
    size_t x_shape_size = 2;
    struct tensor *x = tensor_allocator_alloc(&tensor_alloc, x_shape, x_shape_size, DTYPE);

    size_t y_shape[] = {BATCH_SIZE, 1};
    size_t y_shape_size = 2;
    struct tensor *y_target = tensor_allocator_alloc(&tensor_alloc, y_shape, y_shape_size, DTYPE);
    if (!x || !y_target)
    {
        tensor_allocator_free(&tensor_alloc, x);
        tensor_allocator_free(&tensor_alloc, y_target);
        return EXIT_FAILURE;
    }

    build_example_dataset(x, y_target);

    // Allocate model
    struct linear linear1;
    if (linear_init(&linear1, INPUT_DIM, HIDDEN_DIM, DTYPE, &allocs) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }
    if (linear_xavier_init(&linear1) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    struct linear linear2;
    if (linear_init(&linear2, HIDDEN_DIM, OUT_DIM, DTYPE, &allocs) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }
    if (linear_xavier_init(&linear2) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    // Setup model params
    struct model_params params;
    model_params_init(&params);
    add_model_param(&params, linear1.weight);
    add_model_param(&params, linear1.bias);
    add_model_param(&params, linear2.weight);
    add_model_param(&params, linear2.bias);

    // Setup optimizer
    struct sgd_optimizer opt;
    if (sgd_optimizer_init(&opt, &params, &tensor_alloc) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    double lr = 3e-4;
    double momentum = 0.9;

    size_t epochs = 100;
    for (size_t i = 0; i < epochs; i++)
    {
        // ------------- Forward -------------
        struct tensor *h1 = NULL;
        if (linear_forward(&linear1, x, &h1, intermediates, true) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        struct tensor *h2 = NULL; 
        if (relu_forward(h1, &h2, true, &allocs) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        struct tensor *h3 = NULL;
        if (linear_forward(&linear2, h2, &h3, intermediates, true) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        struct tensor *z = NULL;
        if (mse_loss(h3, y_target, &z, true, &allocs) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        float loss;
        tensor2d_get(z, 0, 0, &loss);
        printf("epoch %ld, loss: %f\n", i, loss);

        // ------------- Backward -------------
        zero_grad(&params);
        backward(z, &allocs);
        sgd_optimizer_step(&opt, lr, momentum, false);

        // Clear iteration allocations
        tensor_list_free_all(intermediates, &tensor_alloc);
        tensor_allocator_free(&tensor_alloc, h2);
        tensor_allocator_free(&tensor_alloc, z);
    }

    // Cleanup
    sgd_optimizer_cleanup(&opt);
    tensor_allocator_free(&tensor_alloc, x);
    tensor_allocator_free(&tensor_alloc, y_target);
    linear_cleanup(&linear1);
    linear_cleanup(&linear2);
    tensor_cpu_allocator_cleanup(&tensor_alloc);
    computational_graph_cpu_allocator_cleanup(&graph_alloc);
    return EXIT_SUCCESS;
}

float compute_example_y_target(float *x_row, float *weight, float bias, size_t dim)
{
    float dot = 0.0;
    for (size_t j = 0; j < dim; j++)
    {
        dot += x_row[j] * weight[j];
    }
    return tanh(dot + bias);
}

void build_example_dataset(struct tensor *x, struct tensor *y_target)
{
    // Random weights and bias for generating y
    float lb = -20;
    float ub = 20;
    float weight[x->shape[1]];
    for (size_t j = 0; j < x->shape[1]; j++)
    {
        weight[j] = sample_uniform(lb, ub);
    }

    float bias = sample_uniform(lb, ub);

    // Populate x with random values and compute y
    for (size_t i = 0; i < x->shape[0]; i++)
    {
        float x_row[x->shape[1]];
        for (size_t j = 0; j < x->shape[1]; j++)
        {
            float value = sample_uniform(lb, ub);
            x_row[j] = value;
            tensor2d_set(x, i, j, value);
        }

        float y_value = compute_example_y_target(x_row, weight, bias, x->shape[1]);
        tensor2d_set(y_target, i, 0, y_value);
    }
}