#include "linear.h"
#include "mse.h"
#include "computational_graph.h"
#include "backpropagation.h"
#include "model_params.h"
#include "tensor.h"
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
    size_t epochs = 10;
    for (size_t i = 0; i < epochs; i++)
    {
        tensor *h1 = tensor2d_alloc(batch_size, out_dim);
        tensor *mult = tensor2d_alloc(batch_size, out_dim);
        linear_forward_graph(x, linear1, mult, h1);

        tensor *z = tensor2d_alloc(1, 1);
        mse_loss_graph(h1, y_target, z);

        // print_computational_graph_node(x->node);
        // print_computational_graph_node(linear1->weights->node);
        // print_computational_graph_node(x->node->parents[0]);
        // print_computational_graph_node(linear1->biases->node);
        // print_computational_graph_node(h1->node);
        // print_computational_graph_node(z->node);
        // printf("\n\n");



        printf("z: ");
        print_tensor(z);

        // backpropagate(&targets);
        zero_grad(&params);        
        print_tensor(x->grad);
        print_tensor(linear1->weights->grad);
        // print_tensor(mult->grad);
        print_tensor(linear1->biases->grad);
        print_tensor(h1->grad);
        print_tensor(z->grad);
        printf("\n\n");

        backward(z, false);

        print_tensor(x->grad);
        print_tensor(linear1->weights->grad);
        // print_tensor(mult->grad);
        print_tensor(linear1->biases->grad);
        print_tensor(h1->grad);
        print_tensor(z->grad);
        printf("\n\n");

        sgd_step(0.00001, &params);

        tensor_free(h1);
        // tensor_free(mult);
        tensor_free(z);

        assert(!x->node && !h1->node && !linear1->weights->node && !linear1->biases->node && !y_target->node && !z->node);
    }

    // print_tensor(linear1->weights);
    // print_tensor(linear1->biases);

    // Cleanup
    tensor_free(x);
    tensor_free(y_target);
    linear_free(linear1);
    return 0;
}