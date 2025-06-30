#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/cross_entropy.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "optimizers/sgd.h"
#include "utils/random.h"
#include "dataset/csv_dataset.h"
#include "dataset/indexes_permutation.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define OUTPUT_ITERATION_FREQ 25

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Wrong number of parameters. Usage:\n %s <mnist_train_dataset_path>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int SEED = 42;
    init_random_seed(SEED);

    const size_t batch_size = 64;
    const size_t input_dim = 784;
    const size_t hidden_dim = 512;
    const size_t num_classes = 10;

    // Can be downloaded from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    struct csv_dataset *train_set = csv_dataset_alloc(argv[1]);
    if (!train_set)
    {
        fprintf(stderr, "Error while trying to open %s.\n", argv[1]);
        return EXIT_FAILURE;
    }

    if (csv_dataset_standard_scale(train_set) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    // Allocate model
    struct linear_layer *linear1 = linear_create(input_dim, hidden_dim);
    if (!linear1)
    {
        return EXIT_FAILURE;
    }
    linear_xavier_init(linear1);

    struct linear_layer *linear2 = linear_create(hidden_dim, num_classes);
    if (!linear2)
    {
        return EXIT_FAILURE;
    }
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
    {
        return EXIT_FAILURE;
    }

    double lr = 3e-4;
    double momentum = 0.9;

    // Setup indexes batch container. In this case, the container's capacity is the batch size.
    struct indexes_batch *ixs_batch = indexes_batch_alloc(batch_size);

    size_t epochs = 2;
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        struct indexes_permutation *permutation = indexes_permutation_alloc(train_set->rows);
        indexes_permutation_init(permutation);

        size_t iteration = 0;
        while (!index_permutation_is_terminated(permutation))
        {
            /***
             * Compute the effective iteration batch size.
             * At each iteration, it represents the effective number of samples sampled from the
             * train set. It handles the case in which we may request to sample 64 samples
             * but only, for instance, 30 remains.
             */
            size_t remaining = index_permutation_get_remaining(permutation);
            size_t iter_batch_size = remaining < batch_size ? remaining : batch_size;

            struct tensor *x = tensor2d_alloc(iter_batch_size, input_dim);
            struct tensor *y = tensor2d_alloc(iter_batch_size, 1);
            if (!x || !y)
            {
                return EXIT_FAILURE;
            }

            // Sample batch indeces
            if (indexes_permutation_sample_index_batch(permutation, ixs_batch, iter_batch_size) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            // Sample batch
            if (csv_dataset_sample_batch(train_set, x, y, ixs_batch) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            // ------------- Forward -------------

            // Linear 1
            struct tensor *mult1 = tensor2d_alloc(iter_batch_size, hidden_dim);
            struct tensor *h1 = tensor2d_alloc(iter_batch_size, hidden_dim);
            if (linear_forward_graph(x, linear1, mult1, h1) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            // ReLU 1
            struct tensor *h2 = tensor2d_alloc(iter_batch_size, hidden_dim);
            if (relu_forward_graph(h1, h2) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            // Linear 2
            struct tensor *mult3 = tensor2d_alloc(iter_batch_size, num_classes);
            struct tensor *h3 = tensor2d_alloc(iter_batch_size, num_classes);
            if (linear_forward_graph(h2, linear2, mult3, h3) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *z = tensor2d_alloc(1, 1);
            if (cross_entropy_loss_graph(h3, y, z) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

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

            index_permutation_update(permutation, iter_batch_size);
            iteration++;
        }
    }

    // Cleanup
    free_sgd_state_tensors(&opt_state);
    linear_free(linear1);
    linear_free(linear2);
    indexes_batch_free(ixs_batch);
    return EXIT_SUCCESS;
}