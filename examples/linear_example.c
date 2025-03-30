#include "linear.h"
#include "mse.h"
#include "computational_graph.h"
#include "backpropagation.h"
#include "model_params.h"
#include "tensor.h"
#include "tensor2d_mult.h"
#include "tensor2d_add_row_vector.h"
#include "sgd.h"
#include "random.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main()
{
    init_random();

    const size_t batch_size = 4;

    // Allocate tensors x (128x4) and y (128x1)
    tensor *x = tensor2d_alloc(batch_size, 4);
    tensor *y_target = tensor2d_alloc(batch_size, 1);
    // if (!x || !y_target)
    // {
    //     tensor_free(x); // Handles NULL gracefully
    //     tensor_free(y_target);
    //     return 1; // Allocation failed
    // }

    // Example: Define weights for the linear relationship (y = x0*1 + x1*2 + x2*3 + x3*4)
    const double weights[] = {1.0, 2.0, 3.0, 4.0};

    // Populate x with sample data and compute y
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            // Example: Fill x with values (e.g., sequential data)
            double value = (double)i + (double)j; // Replace with your data
            tensor2d_set_unchecked(x, i, j, value);
        }

        // Compute y[i] as a linear combination of x[i][0..3] and weights
        double y_value = 0.0;
        for (size_t j = 0; j < 4; j++)
        {
            y_value += x->data[i * x->shape[1] + j] * weights[j];
        }
        tensor2d_set_unchecked(y_target, i, 0, y_value);
    }

    printf("x:\n");
    print_tensor(x);
    printf("y_target:\n");
    print_tensor(y_target);
    // Optional: Verify results
    // print_tensor(x);
    // print_tensor(y);

    size_t in_dim = 4;
    size_t out_dim = 1;
    linear_layer *linear1 = linear_create(in_dim, out_dim);
    linear_xavier_init(linear1);

    // print_tensor(linear1->weights);

    // TODO work on a better interface
    model_params params;
    params.size = 0;
    add_param(&params, linear1->weights);
    add_param(&params, linear1->biases);

    // size_t epochs = 10000000;
    size_t epochs = 10000;
    for (size_t i = 0; i < epochs; i++)
    {
        tensor *mult = tensor2d_alloc(batch_size, out_dim);
        tensor2d_mult_graph(x, linear1->weights, mult);

        tensor *h1 = tensor2d_alloc(batch_size, out_dim);
        tensor2d_add_row_vector_graph(mult, linear1->biases, h1);

        tensor *z = tensor2d_alloc(1, 1);
        mse_loss_graph(h1, y_target, z);

        printf("z: ");
        print_tensor(z);

        zero_grad(&params);        
        backward(z, false);

        sgd_step(0.00001, &params);

        tensor_free(h1);
        tensor_free(mult);
        tensor_free(z);

        assert(!x->node && !linear1->weights->node && !linear1->biases->node && !y_target->node && !z->node && !mult->node);
    }

    // print_tensor(linear1->weights);
    // print_tensor(linear1->biases);

    // Cleanup
    tensor_free(x);
    tensor_free(y_target);
    linear_free(linear1);
    return 0;
}