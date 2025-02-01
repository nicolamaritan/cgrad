#include "mse.h"
#include <stdlib.h>
#include <stdio.h>

void mse_loss(const tensor* const y_pred, const tensor* const y_target, tensor* const z)
{
    double batch_size = y_pred->shape[0];
    z->data[0] = 0;
    //tensor* z = tensor2d_alloc(1, 1);
    for (size_t i = 0; i < batch_size; i++)
    {
        // Compute sample squared error and sum it
        double difference = y_pred->data[i] - y_target->data[i];
        z->data[0] += (0.5 * difference * difference);
    }
    z->data[0] /= batch_size;
}

void mse_loss_graph(tensor* const y_pred, tensor* const y_target, tensor* const z)
{
    mse_loss(y_pred, y_target, z);

    computational_graph_node* y_pred_node = y_pred->node ? y_pred->node : computational_graph_node_tensor_alloc(y_pred);
    computational_graph_node* y_target_node = y_target->node ? y_target->node : computational_graph_node_tensor_alloc(y_target);

    y_target_node->t = (tensor*)y_target;

    computational_graph_node* z_node = computational_graph_node_tensor_alloc(z);
    tensor* one = tensor2d_alloc(1, 1);
    one->data[0] = 1;
    z->grad = one;   

    // Setup connections
    add_parent(y_pred_node, z_node, MSE_PREDICTED);
    add_parent(y_target_node, z_node, MSE_TARGET);
    add_child(z_node, y_pred_node);
    add_child(z_node, y_target_node);

    backpropagation_function_data* data = malloc(sizeof(backpropagation_function_data));
    
    // Setup inputs, in this case composed by predicted and target inside mse_inputs
    mse_inputs* inputs = (mse_inputs*)malloc(sizeof(mse_inputs));   // Maybe struct dependent free after backprop? LOL so much memory leaks TODO
    inputs->predicted = (tensor*)y_pred;
    inputs->target = (tensor*)y_target;
    data->inputs = (void*)inputs;
    z_node->data = data;

    backpropagation_function function = (backpropagation_function)&mse_loss_backpropagate;
    z_node->function = function;

    z_node->free_data = (backpropagation_function_data_cleanup)&free_mse_backpropagation_function_data;
}

tensor* mse_loss_backpropagate(const backpropagation_function_data* const data, const tensor* const D, size_t operand)
{
    mse_inputs* input = (mse_inputs*)data->inputs;

    double batch_size = input->target->shape[0];
    tensor* out = tensor2d_alloc(batch_size, 1);
    for (size_t i = 0; i < batch_size; i++)
    {
        out->data[i] = (input->predicted->data[i] - input->target->data[i]) / batch_size;
    }

    if (operand == MSE_TARGET)
    {
        // Gradient is the same but mult by -1
        for (size_t i = 0; i < out->shape[0]; i++)
        {
            out->data[i] *= -1;
        }
    }

    return out;
}

void free_mse_backpropagation_function_data(backpropagation_function_data* data)
{
    // data->inputs points to a mse_inputs allocation, so we free it but not the prediction and target tensors
    free(data->inputs);    
}