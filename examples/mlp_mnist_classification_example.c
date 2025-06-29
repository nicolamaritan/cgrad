#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/cross_entropy.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "optimizers/sgd.h"
#include "utils/random.h"
#include "dataset/csv_dataset.h"
#include "dataset/index_permutation.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define OUTPUT_ITERATION_FREQ 25

int main()
{
    const int SEED = 42;
    init_random_seed(SEED);

    const size_t batch_size = 64;
    const size_t input_dim = 784;
    const size_t hidden_dim = 512;
    const size_t num_classes = 10;

    // Can be downloaded from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    csv_dataset *train_set = csv_dataset_alloc("./examples/mnist_train.csv");
    if (csv_dataset_standard_scale(train_set) != NO_ERROR)
        exit(1);

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

    size_t epochs = 2;
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        index_permutation *permutation = index_permutation_alloc(train_set->rows);
        index_permutation_init(permutation);
        
        size_t iteration = 0;
        while (!index_permutation_is_terminated(permutation))
        {
            tensor *x = tensor2d_alloc(batch_size, input_dim);
            tensor *y = tensor2d_alloc(batch_size, 1);
            if (!x || !y)
                exit(1); 

            // Sample batch indeces
            index_batch *ix_batch = index_batch_alloc(batch_size);
            if (index_permutation_sample_index_batch(permutation, ix_batch, batch_size) != NO_ERROR)
                exit(1);

            // Sample batch
            if (csv_dataset_sample_batch(train_set, x, y, ix_batch) != NO_ERROR)
                exit(1);

            // ------------- Forward -------------

            // Linear 1
            tensor *mult1 = tensor2d_alloc(batch_size, hidden_dim);
            tensor *h1 = tensor2d_alloc(batch_size, hidden_dim);
            if (linear_forward_graph(x, linear1, mult1, h1) != NO_ERROR)
                exit(1);

            // ReLU 1
            tensor *h2 = tensor2d_alloc(batch_size, hidden_dim);
            if (relu_forward_graph(h1, h2) != NO_ERROR)
                exit(1);

            // Linear 2
            tensor *mult3 = tensor2d_alloc(batch_size, num_classes);
            tensor *h3 = tensor2d_alloc(batch_size, num_classes);
            if (linear_forward_graph(h2, linear2, mult3, h3) != NO_ERROR)
                exit(1);

            tensor *z = tensor2d_alloc(1, 1);
            if (cross_entropy_loss_graph(h3, y, z) != NO_ERROR)
                exit(1);

            if (iteration % OUTPUT_ITERATION_FREQ == 0)
            {
                printf("epoch %02ld, iteration %04ld - loss: %f\n", epoch, iteration, z->data[0]);
            }

            // ------------- Backward -------------
            zero_grad(&params);        
            backward(z, false);

            sgd_step(lr, momentum, false, &opt_state, &params);
            
            // Clear iteration allocations
            tensor_free(x);
            tensor_free(y);
            tensor_free(h1);
            tensor_free(mult1);
            tensor_free(h2);
            tensor_free(h3);
            tensor_free(mult3);
            tensor_free(z);
            index_batch_free(ix_batch);

            index_permutation_update(permutation, batch_size);
            iteration++;
        }
    }

    // Cleanup
    free_sgd_state_tensors(&opt_state);
    linear_free(linear1);
    linear_free(linear2);
    return 0;
}