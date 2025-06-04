#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/cross_entropy.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "optimizers/sgd.h"
#include "utils/random.h"
#include "dataset/csv_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main()
{
    init_random();

    const size_t batch_size = 32;
    const size_t input_dim = 784;
    const size_t hidden_dim = 512;
    const size_t num_classes = 10;

    csv_dataset *train_set = csv_dataset_alloc("./examples/mnist_train.csv");

    for (size_t i = 0; i < train_set->cols; i++)
    {
        printf("%f,", train_set->csv_data[i]);
    }

    return 0;

    tensor *x = tensor2d_alloc(batch_size, input_dim);
    tensor *y_target = tensor2d_alloc(batch_size, 1);
    if (!x || !y_target) {
        tensor_free(x); 
        tensor_free(y_target);
        return 1; 
    }


    // Allocate model
    linear_layer *linear1 = linear_create(input_dim, hidden_dim);
    linear_xavier_init(linear1);

    linear_layer *linear2 = linear_create(hidden_dim, num_classes);
    linear_xavier_init(linear2);

    // Setup model params
    model_params params;
    init_model_params(&params);
    add_model_param(&params, linear1->weights);
    add_model_param(&params, linear1->biases);
    add_model_param(&params, linear2->weights);
    add_model_param(&params, linear2->biases);

    // Setup optimizer
    sgd_state opt_state;
    if (init_sgd_state(&opt_state, &params) != NO_ERROR)
        exit(1);

    double lr = 3e-4;
    double momentum = 0.9;

    size_t epochs = 100;
    for (size_t i = 0; i < epochs; i++)
    {
        // ------------- Forward -------------
        tensor *mult1 = tensor2d_alloc(batch_size, hidden_dim);
        tensor *h1 = tensor2d_alloc(batch_size, hidden_dim);
        if (linear_forward_graph(x, linear1, mult1, h1) != NO_ERROR)
            exit(1);

        tensor *h2 = tensor2d_alloc(batch_size, hidden_dim);
        relu_forward_graph(h1, h2); 

        tensor *mult3 = tensor2d_alloc(batch_size, num_classes);
        tensor *h3 = tensor2d_alloc(batch_size, num_classes);
        if (linear_forward_graph(h2, linear2, mult3, h3) != NO_ERROR)
            exit(1);

        tensor *z = tensor2d_alloc(1, 1);
        if (cross_entropy_loss_graph(h3, y_target, z) != NO_ERROR)
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