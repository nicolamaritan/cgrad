#include "layers/linear.h"
#include "layers/conv2d.h"
#include "layers/relu.h"
#include "losses/cross_entropy.h"
#include "autograd/backpropagation/backpropagation.h"
#include "memory/allocators.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor_reshape.h"
#include "tensor/tensor_print_shape.h"
#include "tensor/tensor_get.h"
#include "optimizers/sgd.h"
#include "dataset/csv_dataset.h"
#include "dataset/indexes_permutation.h"
#include "memory/tensor/cpu/tensor_cpu_allocator.h"
#include "memory/computational_graph/computational_graph_cpu_allocator.h"
#include "utils/random.h"
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

    const size_t batch_size = 64;
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
    struct conv2d conv1;
    if (conv2d_init(&conv1, 1, 4, 3, DTYPE, &tensor_alloc, &allocs) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }
    if (conv2d_xavier_init(&conv1) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    struct conv2d conv2;
    if (conv2d_init(&conv2, 4, 4, 3, DTYPE, &tensor_alloc, &allocs) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }
    if (conv2d_xavier_init(&conv2) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    struct linear linear1;
    if (linear_init(&linear1, 2304, num_classes, DTYPE, &tensor_alloc, &allocs) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }
    if (linear_xavier_init(&linear1) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }


    // Setup model params
    struct model_params params;
    model_params_init(&params);
    add_model_param(&params, conv1.weight);
    add_model_param(&params, conv2.weight);
    add_model_param(&params, linear1.weights);

    // Setup optimizer
    struct sgd_optimizer opt;
    if (sgd_optimizer_init(&opt, &params, &tensor_alloc) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    double lr = 3e-4;
    double momentum = 0.9;

    // Setup indexes batch container. In this case, the container's capacity is the batch size.
    struct indexes_batch *ixs_batch = indexes_batch_alloc(batch_size);
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
            size_t remaining = index_permutation_get_remaining(permutation);
            size_t iter_batch_size = remaining < batch_size ? remaining : batch_size;

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
            struct tensor *x_reshaped = NULL;
            size_t img_shape[] = {batch_size, 1, 28, 28};
            size_t img_shape_size = 4;
            if (tensor_reshape(x, img_shape, img_shape_size, &x_reshaped, true, &allocs) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *h1 = NULL;
            if (conv2d_forward(&conv1, x_reshaped, &h1, intermediates, true) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *h2 = NULL;
            if (relu_forward(h1, &h2, true, &allocs) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *h3 = NULL;
            if (conv2d_forward(&conv2, h2, &h3, intermediates, true) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *h3_flattened = NULL;
            size_t h3_flattened_shape[] = {batch_size, 2304};
            if (tensor_reshape(h3, h3_flattened_shape, 2, &h3_flattened, true, &allocs) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *h4 = NULL;
            if (linear_forward(&linear1, h3_flattened, &h4, intermediates, true) != NO_ERROR)
            {
                return EXIT_FAILURE;
            }

            struct tensor *z = NULL;
            if (cross_entropy_loss(h4, y, &z, true, &allocs) != NO_ERROR)
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
            tensor_allocator_free(&tensor_alloc, x_reshaped);
            tensor_allocator_free(&tensor_alloc, y);
            tensor_allocator_free(&tensor_alloc, h1);
            tensor_allocator_free(&tensor_alloc, h2);
            tensor_allocator_free(&tensor_alloc, h3);
            tensor_allocator_free(&tensor_alloc, h3_flattened);
            tensor_allocator_free(&tensor_alloc, h4);
            tensor_allocator_free(&tensor_alloc, z);
            intermediates->size = 0;

            index_permutation_update(permutation, iter_batch_size);
            iteration++;
        }
    }

    // Cleanup
    sgd_optimizer_cleanup(&opt);
    conv2d_cleanup(&conv1);
    conv2d_cleanup(&conv2);
    indexes_batch_free(ixs_batch);
    tensor_cpu_allocator_cleanup(&tensor_alloc);
    computational_graph_cpu_allocator_cleanup(&graph_alloc);
    return EXIT_SUCCESS;
}