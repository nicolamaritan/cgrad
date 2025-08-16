#include "memory/allocators.h"
#include "memory/tensor/cpu/tensor_cpu_allocator.h"
#include "memory/computational_graph/computational_graph_cpu_allocator.h"
#include "datastructures/tensor_list.h"
#include "layers/relu.h"
#include "losses/mse.h"
#include "autograd/backpropagation/backpropagation.h"
#include "memory/allocators.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor_get.h"
#include "tensor/tensor_set.h"
#include "optimizers/sgd.h"
#include "layers/conv2d.h"
#include "tensor/tensor_helpers.h"
#include "tensor/tensor_reshape.h"
#include "utils/random.h"
#include <stdio.h>

int main()
{
    const int SEED = 42;
    init_random_seed(SEED);

    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    // Allocator initialization
    struct tensor_allocator tensor_alloc;
    tensor_cpu_allocator_init(&tensor_alloc);

    struct computational_graph_allocator graph_alloc;
    computational_graph_cpu_allocator_init(&graph_alloc);

    struct allocators allocs = {&tensor_alloc, &graph_alloc};

    size_t x_shape[] = {2, 3, 4, 4};
    size_t x_shape_size = 4;
    struct tensor *x = tensor_allocator_alloc(&tensor_alloc, x_shape, x_shape_size, DTYPE);
    float *x_data = (float *)x->data;
    for (int i = 0; i < 16 * 3; i++)
        x_data[i] = (i % 16) + 1;
    for (int i = 0; i < 16 * 3; i++)
        x_data[i + x->stride[0]] = 3 * ((i % 16) + 1);

    struct conv2d conv_1;
    conv2d_init(&conv_1, 3, 2, 2, x->dtype, &tensor_alloc, &allocs);
    conv2d_xavier_init(&conv_1);

    const size_t INTERMEDIATES_CAPACITY = 20;
    struct tensor_list *intermediates = tensor_list_alloc(INTERMEDIATES_CAPACITY);

    // size_t k_shape[] = {2, 3, 2, 2};
    // size_t k_shape_size = 4;
    // struct tensor *k = tensor_allocator_alloc(&tensor_alloc, k_shape, k_shape_size, DTYPE);
    // float *k_data = (float *)k->data;
    // k_data[0] = 1;
    // k_data[1] = 2;
    // k_data[2] = 1;
    // k_data[3] = 1;

    // k_data[4] = 1;
    // k_data[5] = 2;
    // k_data[6] = 1;
    // k_data[7] = 1;

    // k_data[8] = 1;
    // k_data[9] = 2;
    // k_data[10] = 1;
    // k_data[11] = 1;

    // k_data[12] = 2;
    // k_data[13] = 4;
    // k_data[14] = 2;
    // k_data[15] = 2;

    // k_data[16] = 2;
    // k_data[17] = 4;
    // k_data[18] = 2;
    // k_data[19] = 2;

    // k_data[20] = 2;
    // k_data[21] = 4;
    // k_data[22] = 2;
    // k_data[23] = 2;

    // conv_1.weight = k;


    // Setup model params
    struct model_params params;
    model_params_init(&params);
    add_model_param(&params, conv_1.weight);

    // Setup optimizer
    struct sgd_optimizer opt;
    if (sgd_optimizer_init(&opt, &params, &tensor_alloc) != NO_ERROR)
    {
        return EXIT_FAILURE;
    }

    double lr = 3e-5;
    double momentum = 0.9;

    size_t y_shape[] = {2, 1};
    size_t y_shape_size = 2;
    struct tensor *y_target = tensor_allocator_alloc(&tensor_alloc, y_shape, y_shape_size, DTYPE);
    float* y_data = y_target->data;
    y_data[0] = 0;
    y_data[1] = 0;

    size_t dp_shape[] = {18, 1};
    struct tensor *down_proj = tensor_allocator_alloc(&tensor_alloc, dp_shape, 2, DTYPE);
    float *dp_data = down_proj->data;
    for (size_t i = 0; i < down_proj->shape_size; i++)
    {
        dp_data[i] = 1.0;
    }

    // Prepare output
    for (size_t epoch = 0; epoch < 100; epoch++)
    {
        // Run convolution
        struct tensor *h1 = NULL;
        cgrad_error err = conv2d_forward(&conv_1, x, &h1, intermediates, true);
        if (err != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        struct tensor *flat = NULL;
        size_t shape[] = {h1->shape[0], h1->shape[1] * h1->shape[2] * h1->shape[3]};
        if (tensor_reshape(h1, shape, 2, &flat, true, &allocs) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        struct tensor *down_projected = NULL;
        tensor2d_mult(flat, down_proj, &down_projected, true, &allocs);

        struct tensor *z = NULL;
        if (mse_loss(down_projected, y_target, &z, true, &allocs) != NO_ERROR)
        {
            return EXIT_FAILURE;
        }

        float loss;
        tensor2d_get(z, 0, 0, &loss);
        printf("epoch %ld, loss: %f\n", epoch, loss);

        // ------------- Backward -------------
        zero_grad(&params);
        backward(z, &allocs);
        sgd_optimizer_step(&opt, lr, momentum, false);

        // Clear iteration allocations
        tensor_list_free_all(intermediates, &tensor_alloc);
        tensor_allocator_free(&tensor_alloc, flat);
        tensor_allocator_free(&tensor_alloc, down_projected);
        tensor_allocator_free(&tensor_alloc, z);
    }

    // Print output
    // struct tensor *out = conv1_out.result;
    // float *out_data = (float *)out->data;
    // size_t H_out = out->shape[2];
    // size_t W_out = out->shape[3];

    // printf("Output tensor:\n");
    // for (size_t b = 0; b < out->shape[0]; b++)
    // {
    //     for (size_t c = 0; c < out->shape[1]; c++)
    //     {
    //         for (size_t h = 0; h < H_out; h++)
    //         {
    //             for (size_t w = 0; w < W_out; w++)
    //             {
    //                 printf("%6.1f ", out_data[b * out->stride[0] + c * (W_out * H_out) + h * W_out + w]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}