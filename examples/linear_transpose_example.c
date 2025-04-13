#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/mse.h"
#include "autograd/backpropagation.h"
#include "model/model_params.h"
#include "tensor/tensor.h"
#include "tensor/tensor2d_mult.h"
#include "tensor/tensor2d_add_row_vector.h"
#include "tensor/tensor2d_trans.h"
#include "optimizers/sgd.h"
#include "utils/random.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main()
{
    init_random();

    const size_t batch_size = 128;

    // Allocate tensors x (128x4) and y (128x1)
    tensor *x = tensor2d_alloc(batch_size, 4);
    tensor *y_target = tensor2d_alloc(batch_size, 1);

    // Example: Define weights for the linear relationship (y = x0*1 + x1*2 + x2*3 + x3*4)
    const double true_weights[] = {1.0, 2.0, 3.0, 4.0};

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
            y_value += x->data[i * x->shape[1] + j] * true_weights[j];
        }
        tensor2d_set_unchecked(y_target, i, 0, y_value);
    }

    size_t in_dim = 4;
    size_t out_dim = 1;

    // Extract weights
    linear_layer *linear1 = linear_create(out_dim, in_dim); // Dimensions swapped. Requires transpose
    linear_xavier_init(linear1);
    tensor *weights = linear1->weights;

    model_params params;
    params.size = 0;
    add_model_param(&params, weights);

    sgd_state opt_state;
    if (init_sgd_state(&opt_state, &params) != NO_ERROR)
        exit(1);

    size_t epochs = 10000;
    for (size_t i = 0; i < epochs; i++)
    {
        tensor *weights_transposed = tensor2d_alloc(in_dim, out_dim);
        if (tensor2d_trans_graph(weights, weights_transposed) != NO_ERROR)
            exit(1);

        tensor *mult = tensor2d_alloc(batch_size, out_dim);
        if (tensor2d_mult_graph(x, weights_transposed, mult) != NO_ERROR)
            exit(1);

        tensor *z = tensor2d_alloc(1, 1);
        mse_loss_graph(mult, y_target, z);

        printf("z: ");
        print_tensor(z);

        zero_grad(&params);
        backward(z, false);

        sgd_step(0.00001, 0.9, false, &opt_state, &params);

        tensor_free(mult);
        tensor_free(z);

        assert(!x->node && !linear1->weights->node && !linear1->biases->node && !y_target->node && !z->node && !mult->node);
    }

    // Cleanup
    tensor_free(x);
    tensor_free(y_target);
    linear_free(linear1);
    return 0;
}