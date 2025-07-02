#include "losses/mse.h"
#include <stdlib.h>
#include <stdio.h>

cgrad_error mse_loss(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z)
{
    if (!y_pred || !y_target || !z)
    {
        return TENSOR_NULL;
    }
    if (!y_pred->data || !y_target->data || !z->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (!y_pred->shape || !y_target->shape || !z->shape)
    {
        return TENSOR_SHAPE_NULL;
    }
    if (y_pred->data_size != y_target->data_size)
    {
        return TENSOR_DATA_SIZE_MISMATCH;
    }
    if (!tensor_same_shape(y_pred, y_target))
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    double batch_size = y_pred->shape[0];
    z->data[0] = 0;

    for (size_t i = 0; i < batch_size; i++)
    {
        // Compute sample squared error and sum it
        double difference = y_pred->data[i] - y_target->data[i];
        z->data[0] += (0.5 * difference * difference);
    }
    z->data[0] /= batch_size;

    return NO_ERROR;
}

cgrad_error mse_loss_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor *const z)
{
    cgrad_error err = mse_loss(y_pred, y_target, z);
    if (err != NO_ERROR)
    {
        return err;
    }

    add_computational_graph_link(y_pred, MSE_PREDICTED, z, &mse_loss_backpropagate_predicted);
    add_computational_graph_link(y_target, MSE_TARGET, z, &mse_loss_backpropagate_target);

    return NO_ERROR;
}

void mse_loss_backpropagate_predicted(const struct tensor **const operands, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    const struct tensor *predicted = operands[MSE_PREDICTED];
    const struct tensor *target= operands[MSE_TARGET];
    double batch_size = target->shape[0];
    for (size_t i = 0; i < batch_size; i++)
    {
        grad_wrt_operand->data[i] = (predicted->data[i] - target->data[i]) / batch_size;
    }
}

void mse_loss_backpropagate_target(const struct tensor **const operands, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    mse_loss_backpropagate_predicted(operands, grad_wrt_out, grad_wrt_operand);

    // Gradient is the same but mult by -1
    for (size_t i = 0; i < grad_wrt_operand->shape[0]; i++)
    {
        grad_wrt_operand->data[i] *= -1;
    }
}