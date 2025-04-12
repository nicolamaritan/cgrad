#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/mse.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_add_row_vector.h"
#include "optimizers/sgd.h"
#include "utils/random.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    const size_t batch_size = 128;

    // Allocate tensors x (128x4) and y (128x1)
    tensor *x = tensor2d_alloc(batch_size, 4);
    tensor *y_target = tensor2d_alloc(batch_size, 1);
    if (!x || !y_target)
    {
        tensor_free(x); // Handles NULL gracefully
        tensor_free(y_target);
        return 1; // Allocation failed
    }

    // Example: Define weights for the linear relationship (y = x0*1 + x1*2 + x2*3 + x3*4)
    const double weights[] = {1.0, 2.0, 3.0, 4.0};

    double negative_offset = 50.0;
    // Populate x with sample data and compute y
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            // Example: Fill x with values (e.g., sequential data)
            double value = (double)i + (double)j - negative_offset; // Replace with your data
            tensor2d_set_unchecked(x, i, j, value);
        }

        // Compute y[i] as a linear combination of x[i][0..3] and weights
        double y_value = 0.0;
        for (size_t j = 0; j < 4; j++)
        {
            y_value += x->data[i * x->shape[1] + j] * weights[j];
        }
        tensor2d_set_unchecked(y_target, i, 0, fmax(y_value, 0));
    }

    // Optional: Verify results
    // print_tensor(x);
    // print_tensor(y);

    init_random();

    size_t in_dim = 4;
    size_t out_dim = 1;
    linear_layer *linear1 = linear_create(in_dim, out_dim);
    linear_xavier_init(linear1);

    // TODO work on a better interface
    model_params params;
    params.size = 0;
    add_param(&params, linear1->weights);
    add_param(&params, linear1->biases);

    size_t epochs = 1000;
    for (size_t i = 0; i < epochs; i++)
    {
        tensor *h1 = tensor2d_alloc(batch_size, out_dim);
        tensor *mult= tensor2d_alloc(batch_size, out_dim);
        linear_forward_graph(x, linear1, mult, h1);
        // printf("h1: ");
        // print_tensor(h1);
        // printf("\n\n");

        tensor *h2 = tensor2d_alloc(batch_size, out_dim);
        cgrad_error err = relu_forward_graph(h1, h2);
        if (err != NO_ERROR)
        {
            fprintf(stderr, "Error: %d.\n", err);
            exit(1);
        }
        
        // printf("h2: ");
        // print_tensor(h2);
        // printf("\n\n");

        tensor *z = tensor2d_alloc(1, 1);
        mse_loss_graph(h2, y_target, z);

        printf("z: ");
        print_tensor(z);
        printf("\n\n");

        zero_grad(&params);        
        backward(z, false);

        if (i + 1 == epochs)
        {
            printf("Gradients:\n");
            print_tensor(mult->grad);
            printf("\n");
            print_tensor(h1->grad);
            printf("\n");
            print_tensor(h2->grad);
            printf("\n");
            print_tensor(z->grad);
        }

        sgd_step(0.00001, &params);

        free(linear1->weights->node);
        linear1->weights->node = NULL;
        free(linear1->biases->node);
        linear1->biases->node = NULL;

        free(x->node);
        x->node = NULL;
        free(y_target->node);
        y_target->node = NULL;
        free(z->node);
        z->node = NULL;
    }

    print_tensor(linear1->weights);
    print_tensor(linear1->biases);

    // Cleanup
    tensor_free(x);
    tensor_free(y_target);
    return 0;
    
   return 0;
}