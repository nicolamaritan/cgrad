#include "cgrad/layers/linear.h"
#include "cgrad/layers/relu.h"
#include "cgrad/losses/cross_entropy.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/memory/allocators.h"
#include "cgrad/model/model_params.h"
#include "cgrad/tensor/tensor.h"
#include "cgrad/tensor/tensor_get.h"
#include "cgrad/optimizers/sgd.h"
#include "cgrad/dataset/csv_dataset.h"
#include "cgrad/dataset/indexes_permutation.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_allocator.h"
#include "cgrad/utils/random.h"
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

    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct computational_graph_cpu_pool graph_pool;
    if (computational_graph_cpu_pool_init(&graph_pool) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    // Allocator initialization
    struct tensor_allocator tensor_alloc;
    tensor_cpu_allocator_init(&tensor_alloc);

    struct computational_graph_allocator graph_alloc;
    computational_graph_cpu_allocator_init(&graph_alloc);

    struct allocators allocs = {&tensor_alloc, &graph_alloc};

    const size_t INTERMEDIATES_CAPACITY = 20;
    struct tensor_list *intermediates = tensor_list_alloc(INTERMEDIATES_CAPACITY);

    const size_t BATCH_SIZE = 64;
    const size_t INPUT_DIM = 784;
    const size_t HIDDEN_DIM = 512;
    const size_t NUM_CLASSES = 10;

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
    if (linear_init(&linear2, HIDDEN_DIM, NUM_CLASSES, DTYPE, &allocs) != NO_ERROR)
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

    // Setup indexes batch container. In this case, the container's capacity is the batch size.
    struct indexes_batch *ixs_batch = indexes_batch_alloc(BATCH_SIZE);
    if (!ixs_batch)
    {
        return EXIT_FAILURE;
    }

    size_t epochs = 1;
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        struct indexes_permutation *permutation = indexes_permutation_alloc(train_set->rows);
        if (!permutation)
        {
            return EXIT_FAILURE;
        }

        if (indexes_permutation_init(permutation) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

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
            size_t iter_batch_size = remaining < BATCH_SIZE ? remaining : BATCH_SIZE;

            // Sample batch indeces
            if (indexes_permutation_sample_index_batch(permutation, ixs_batch, iter_batch_size) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *x = NULL;
            struct tensor *y = NULL;
            // Sample batch
            if (csv_dataset_sample_batch(train_set, &x, &y, ixs_batch, DTYPE, &tensor_alloc) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

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
            if (cross_entropy_loss(h3, y, &z, true, &allocs) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            if (iteration % OUTPUT_ITERATION_FREQ == 0)
            {
                float loss;
                tensor2d_get(z, 0, 0, &loss);
                printf("epoch %02ld, iteration %04ld - loss: %f\n", epoch, iteration, loss);
            }

            // ------------- Backward -------------
            zero_grad(&params);
            backward(z, &allocs);

            sgd_optimizer_step(&opt, lr, momentum, false);

            // Clear iteration allocations
            tensor_list_free_all(intermediates, &tensor_alloc);
            tensor_allocator_free(&tensor_alloc, x);
            tensor_allocator_free(&tensor_alloc, y);
            tensor_allocator_free(&tensor_alloc, h1);
            tensor_allocator_free(&tensor_alloc, h2);
            tensor_allocator_free(&tensor_alloc, h3);
            tensor_allocator_free(&tensor_alloc, z);
            intermediates->size = 0;

            index_permutation_update(permutation, iter_batch_size);
            iteration++;
        }
    }

    // Cleanup
    sgd_optimizer_cleanup(&opt);
    linear_cleanup(&linear1);
    linear_cleanup(&linear2);
    indexes_batch_free(ixs_batch);
    tensor_cpu_allocator_cleanup(&tensor_alloc);
    computational_graph_cpu_allocator_cleanup(&graph_alloc);
    return EXIT_SUCCESS;
}