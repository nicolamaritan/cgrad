#include "loss/mse.h"
#include <stdlib.h>
#include <stdio.h>

cgrad_error mse_loss(const tensor *const y_pred, const tensor *const y_target, tensor *const z)
{
    if (!y_pred || !y_target || !z)
        return TENSOR_NULL;
    if (!y_pred->data || !y_target->data || !z->data)
        return TENSOR_DATA_NULL;
    if (!y_pred->shape || !y_target->shape || !z->shape)
        return TENSOR_SHAPE_NULL;
    if (y_pred->data_size != y_target->data_size)
        return TENSOR_DATA_SIZE_MISMATCH;
    if (!tensor_same_shape(y_pred, y_target))
        return TENSOR_SHAPE_MISMATCH;

    double batch_size = y_pred->shape[0];
    z->data[0] = 0;
    // tensor* z = tensor2d_alloc(1, 1);
    for (size_t i = 0; i < batch_size; i++)
    {
        // Compute sample squared error and sum it
        double difference = y_pred->data[i] - y_target->data[i];
        z->data[0] += (0.5 * difference * difference);
    }
    z->data[0] /= batch_size;

    return NO_ERROR;
}

cgrad_error mse_loss_graph(tensor *const y_pred, tensor *const y_target, tensor *const z)
{
    cgrad_error err = mse_loss(y_pred, y_target, z);
    if (err != NO_ERROR)
        return err;

    computational_graph_node *y_pred_node = y_pred->node ? y_pred->node : computational_graph_node_tensor_alloc(y_pred);
    computational_graph_node *y_target_node = y_target->node ? y_target->node : computational_graph_node_tensor_alloc(y_target);

    y_target_node->t = (tensor *)y_target;

    computational_graph_node *z_node = computational_graph_node_tensor_alloc(z);

    // Setup connections
    add_parent(y_pred_node, z_node, MSE_PREDICTED);
    add_parent(y_target_node, z_node, MSE_TARGET);
    add_child(z_node, y_pred_node);
    add_child(z_node, y_target_node);

    // Setup backpropation functions 
    z_node->function[MSE_PREDICTED] = (backpropagation_function)&mse_loss_backpropagate_predicted;
    z_node->function[MSE_TARGET] = (backpropagation_function)&mse_loss_backpropagate_target;

    // Setup operands
    z_node->tensor_operands[MSE_PREDICTED] = y_pred;
    z_node->tensor_operands[MSE_TARGET] = y_target;

    return NO_ERROR;
}

void mse_loss_backpropagate_predicted(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand)
{
    const tensor *predicted = operands[MSE_PREDICTED];
    const tensor *target= operands[MSE_TARGET];
    double batch_size = target->shape[0];
    for (size_t i = 0; i < batch_size; i++)
    {
        grad_wrt_operand->data[i] = (predicted->data[i] - target->data[i]) / batch_size;
    }
}

void mse_loss_backpropagate_target(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand)
{
    mse_loss_backpropagate_predicted(operands, grad_wrt_out, grad_wrt_operand);

    // Gradient is the same but mult by -1
    for (size_t i = 0; i < grad_wrt_operand->shape[0]; i++)
    {
        grad_wrt_operand->data[i] *= -1;
    }
}