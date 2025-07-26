#include "losses/mse.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <stdlib.h>
#include <stdio.h>

typedef enum mse_loss_operand
{
    MSE_PREDICTED,
    MSE_TARGET
} mse_loss_operand;


static cgrad_error mse_loss_dispatch(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
static void mse_loss_unchecked_f64(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
static void mse_loss_unchecked_f32(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
static void mse_loss_backpropagate_predicted(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void mse_loss_backpropagate_predicted_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void mse_loss_backpropagate_predicted_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void mse_loss_backpropagate_target(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void mse_loss_backpropagate_target_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void mse_loss_backpropagate_target_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

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
    if (y_pred->data_size != y_target->data_size)
    {
        return TENSOR_DATA_SIZE_MISMATCH;
    }
    if (!tensor_same_shape(y_pred, y_target))
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    return mse_loss_dispatch(y_pred, y_target, z);
}

static cgrad_error mse_loss_dispatch(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z)
{
    switch (y_pred->dtype)
    {
        case DTYPE_FLOAT64:
            mse_loss_unchecked_f64(y_pred, y_target, z);
            break;
        case DTYPE_FLOAT32:
            mse_loss_unchecked_f32(y_pred, y_target, z);
            break;
        default:
            return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void mse_loss_unchecked_f64(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z)
{
    double batch_size = y_pred->shape[0];
    double *z_data = (double *)z->data;
    double *y_pred_data = (double *)y_pred->data;
    double *y_target_data = (double *)y_target->data;

    z_data[0] = 0;

    for (size_t i = 0; i < batch_size; i++)
    {
        // Compute sample squared error and sum it
        double difference = y_pred_data[i] - y_target_data[i];
        z_data[0] += (0.5 * difference * difference);
    }
    z_data[0] /= batch_size;
}

static void mse_loss_unchecked_f32(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z)
{
    float batch_size = y_pred->shape[0];
    float *z_data = (float *)z->data;
    float *y_pred_data = (float *)y_pred->data;
    float *y_target_data = (float *)y_target->data;

    z_data[0] = 0;

    for (size_t i = 0; i < batch_size; i++)
    {
        // Compute sample squared error and sum it
        float difference = y_pred_data[i] - y_target_data[i];
        z_data[0] += (0.5 * difference * difference);
    }
    z_data[0] /= batch_size;
}

cgrad_error mse_loss_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor *const z, struct autograd_allocators *ag_allocators)
{
    cgrad_error err = mse_loss(y_pred, y_target, z);
    if (err != NO_ERROR)
    {
        return err;
    }

    add_computational_graph_link(y_pred, MSE_PREDICTED, z, &mse_loss_backpropagate_predicted, ag_allocators);
    add_computational_graph_link(y_target, MSE_TARGET, z, &mse_loss_backpropagate_target, ag_allocators);

    return NO_ERROR;
}

static void mse_loss_backpropagate_predicted(const struct backpropagation_context *const ctx, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
        case DTYPE_FLOAT64:
            mse_loss_backpropagate_predicted_f64(ctx, grad_wrt_out, grad_wrt_operand);
            break;
        case DTYPE_FLOAT32:
            mse_loss_backpropagate_predicted_f32(ctx, grad_wrt_out, grad_wrt_operand);
            break;
        default:
            break;
    }
}

static void mse_loss_backpropagate_predicted_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *predicted = ctx->operands[MSE_PREDICTED];
    const struct tensor *target= ctx->operands[MSE_TARGET];

    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;
    double *predicted_data = (double *)predicted->data;
    double *target_data = (double *)target->data;

    double batch_size = target->shape[0];
    for (size_t i = 0; i < batch_size; i++)
    {
        grad_wrt_operand_data[i] = (predicted_data[i] - target_data[i]) / batch_size;
    }
}

static void mse_loss_backpropagate_predicted_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *predicted = ctx->operands[MSE_PREDICTED];
    const struct tensor *target= ctx->operands[MSE_TARGET];

    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    float *predicted_data = (float *)predicted->data;
    float *target_data = (float *)target->data;

    float batch_size = target->shape[0];
    for (size_t i = 0; i < batch_size; i++)
    {
        grad_wrt_operand_data[i] = (predicted_data[i] - target_data[i]) / batch_size;
    }
}

static void mse_loss_backpropagate_target(const struct backpropagation_context *const ctx, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
        case DTYPE_FLOAT64:
            mse_loss_backpropagate_target_f64(ctx, grad_wrt_out, grad_wrt_operand);
            break;
        case DTYPE_FLOAT32:
            mse_loss_backpropagate_target_f32(ctx, grad_wrt_out, grad_wrt_operand);
            break;
        default:
            break;
    }
}

static void mse_loss_backpropagate_target_f64(const struct backpropagation_context *const ctx, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    mse_loss_backpropagate_predicted_f64(ctx, grad_wrt_out, grad_wrt_operand);

    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;
    // Gradient is the same but mult by -1
    for (size_t i = 0; i < grad_wrt_operand->shape[0]; i++)
    {
        grad_wrt_operand_data[i] *= -1;
    }
}

static void mse_loss_backpropagate_target_f32(const struct backpropagation_context *const ctx, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    mse_loss_backpropagate_predicted_f32(ctx, grad_wrt_out, grad_wrt_operand);

    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    // Gradient is the same but mult by -1
    for (size_t i = 0; i < grad_wrt_operand->shape[0]; i++)
    {
        grad_wrt_operand_data[i] *= -1;
    }
}