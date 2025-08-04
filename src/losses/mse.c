#include "losses/mse.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <stdlib.h>
#include <stdio.h>

typedef enum mse_loss_operand
{
    MSE_PREDICTED,
    MSE_TARGET
} mse_loss_operand;

static inline cgrad_error mse_loss_update_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor **const z, struct allocators *const allocs);
static cgrad_error mse_loss_dispatch(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
static cgrad_error mse_loss_f64(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
static cgrad_error mse_loss_f32(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
static cgrad_error mse_loss_backpropagate_predicted(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error mse_loss_backpropagate_predicted_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error mse_loss_backpropagate_predicted_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error mse_loss_backpropagate_target(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error mse_loss_backpropagate_target_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error mse_loss_backpropagate_target_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error mse_loss(struct tensor *const y_pred, struct tensor *const y_target, struct tensor **const z, const bool track_grad, struct allocators *const allocs)
{
    if (!y_pred || !y_target)
    {
        return TENSOR_NULL;
    }
    if (!y_pred->data || !y_target->data)
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

    const size_t shape[] = {1, 1};
    const size_t shape_size = 2;
    (*z) = tensor_allocator_alloc(allocs->tensor_alloc, shape, shape_size, y_pred->dtype);

    if (!(*z))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    cgrad_error err = mse_loss_dispatch(y_pred, y_target, *z);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return mse_loss_update_graph(y_pred, y_target, z, allocs);
    }

    return NO_ERROR;
}

static inline cgrad_error mse_loss_update_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor **const z, struct allocators *const allocs)
{
    cgrad_error err = add_computational_graph_link(y_pred, MSE_PREDICTED, *z, &mse_loss_backpropagate_predicted, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    return add_computational_graph_link(y_target, MSE_TARGET, *z, &mse_loss_backpropagate_target, allocs);
}

static cgrad_error mse_loss_dispatch(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z)
{
    switch (y_pred->dtype)
    {
    case DTYPE_FLOAT64:
        return mse_loss_f64(y_pred, y_target, z);
    case DTYPE_FLOAT32:
        return mse_loss_f32(y_pred, y_target, z);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error mse_loss_f64(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z)
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

    return NO_ERROR;
}

static cgrad_error mse_loss_f32(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z)
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

    return NO_ERROR;
}

static cgrad_error mse_loss_backpropagate_predicted(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
    case DTYPE_FLOAT64:
        return mse_loss_backpropagate_predicted_f64(ctx, grad_wrt_out, grad_wrt_operand);
    case DTYPE_FLOAT32:
        return mse_loss_backpropagate_predicted_f32(ctx, grad_wrt_out, grad_wrt_operand);
    default:
        return NO_ERROR;
    }
}

static cgrad_error mse_loss_backpropagate_predicted_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *predicted = ctx->operands[MSE_PREDICTED];
    const struct tensor *target = ctx->operands[MSE_TARGET];

    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;
    double *predicted_data = (double *)predicted->data;
    double *target_data = (double *)target->data;

    double batch_size = target->shape[0];
    for (size_t i = 0; i < batch_size; i++)
    {
        grad_wrt_operand_data[i] = (predicted_data[i] - target_data[i]) / batch_size;
    }

    return NO_ERROR;
}

static cgrad_error mse_loss_backpropagate_predicted_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *predicted = ctx->operands[MSE_PREDICTED];
    const struct tensor *target = ctx->operands[MSE_TARGET];

    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    float *predicted_data = (float *)predicted->data;
    float *target_data = (float *)target->data;

    float batch_size = target->shape[0];
    for (size_t i = 0; i < batch_size; i++)
    {
        grad_wrt_operand_data[i] = (predicted_data[i] - target_data[i]) / batch_size;
    }

    return NO_ERROR;
}

static cgrad_error mse_loss_backpropagate_target(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
    case DTYPE_FLOAT64:
        return mse_loss_backpropagate_target_f64(ctx, grad_wrt_out, grad_wrt_operand);
    case DTYPE_FLOAT32:
        return mse_loss_backpropagate_target_f32(ctx, grad_wrt_out, grad_wrt_operand);
    default:
        return NO_ERROR;
    }
}

static cgrad_error mse_loss_backpropagate_target_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    cgrad_error err = NO_ERROR;
    if ((err = mse_loss_backpropagate_predicted_f64(ctx, grad_wrt_out, grad_wrt_operand)) != NO_ERROR)
    {
        return NO_ERROR;
    }

    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;
    // Gradient is the same but mult by -1
    for (size_t i = 0; i < grad_wrt_operand->shape[0]; i++)
    {
        grad_wrt_operand_data[i] *= -1;
    }

    return NO_ERROR;
}

static cgrad_error mse_loss_backpropagate_target_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    cgrad_error err = NO_ERROR;
    if ((err = mse_loss_backpropagate_predicted_f32(ctx, grad_wrt_out, grad_wrt_operand)) != NO_ERROR)
    {
        return NO_ERROR;
    }

    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    // Gradient is the same but mult by -1
    for (size_t i = 0; i < grad_wrt_operand->shape[0]; i++)
    {
        grad_wrt_operand_data[i] *= -1;
    }

    return NO_ERROR;
}