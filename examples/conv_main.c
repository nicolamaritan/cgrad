#include "memory/allocators.h"
#include "memory/tensor/cpu/tensor_cpu_allocator.h"
#include "memory/computational_graph/computational_graph_cpu_allocator.h"
#include "tensor/tensor_conv2d.h"
#include "tensor/tensor_helpers.h"
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

    size_t k_shape[] = {2, 3, 2, 2};
    size_t k_shape_size = 4;
    struct tensor *k = tensor_allocator_alloc(&tensor_alloc, k_shape, k_shape_size, DTYPE);
    float *k_data = (float *)k->data;
    k_data[0] = 1;
    k_data[1] = 2;
    k_data[2] = 1;
    k_data[3] = 1;

    k_data[4] = 1;
    k_data[5] = 2;
    k_data[6] = 1;
    k_data[7] = 1;

    k_data[8] = 1;
    k_data[9] = 2;
    k_data[10] = 1;
    k_data[11] = 1;

    k_data[12] = 2;
    k_data[13] = 4;
    k_data[14] = 2;
    k_data[15] = 2;

    k_data[16] = 2;
    k_data[17] = 4;
    k_data[18] = 2;
    k_data[19] = 2;

    k_data[20] = 2;
    k_data[21] = 4;
    k_data[22] = 2;
    k_data[23] = 2;

    // Prepare output
    struct tensor *out = NULL;

    // Run convolution
    tensor_conv2d(x, k, &out, false, &allocs);

    // Print output
    float *out_data = (float *)out->data;
    size_t H_out = out->shape[2];
    size_t W_out = out->shape[3];

    printf("Output tensor:\n");
    for (size_t b = 0; b < out->shape[0]; b++)
    {
        for (size_t c = 0; c < out->shape[1]; c++)
        {
            for (size_t h = 0; h < H_out; h++)
            {
                for (size_t w = 0; w < W_out; w++)
                {
                    printf("%6.1f ", out_data[b * out->stride[0] + c * (W_out * H_out) + h * W_out + w]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}