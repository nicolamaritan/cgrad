#include "losses/cross_entropy.h"
#include "tensor/tensor_get.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef enum cross_entropy_loss_operand
{
    CROSS_ENTROPY_PREDICTED,
    CROSS_ENTROPY_TARGET
} cross_entropy_loss_operand;

static cgrad_error cross_entropy_loss_dispatch(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss);
static void cross_entropy_loss_unchecked_f64(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss);
static void cross_entropy_loss_unchecked_f32(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss);
static double compute_softmax_normalization_f64(const struct tensor *const logits, const size_t row);
static float compute_softmax_normalization_f32(const struct tensor *const logits, const size_t row);
static void cross_entropy_loss_backpropagate_predicted(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void cross_entropy_loss_backpropagate_predicted_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void cross_entropy_loss_backpropagate_predicted_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error cross_entropy_loss(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss)
{
    if (!logits || !targets || !loss)
    {
        return TENSOR_NULL;
    }
    if (!logits->data || !targets->data || !loss->data)
    {
        return TENSOR_DATA_NULL;
    }
    // TODO add check on tensor shape

    return cross_entropy_loss_dispatch(logits, targets, loss);
}

static cgrad_error cross_entropy_loss_dispatch(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss)
{
    switch (logits->cgrad_dtype)
    {
    case DTYPE_FLOAT64:
        cross_entropy_loss_unchecked_f64(logits, targets, loss);
        break;
    case DTYPE_FLOAT32:
        cross_entropy_loss_unchecked_f32(logits, targets, loss);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void cross_entropy_loss_unchecked_f64(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss)
{
    double batch_size = logits->shape[0];
    double *loss_data = (double *)loss->data;
    loss_data[0] = 0;
    for (size_t i = 0; i < batch_size; i++)
    {
        // Should add support for different data integer
        double target_label_double = 0;
        tensor2d_get(targets, i, 0, &target_label_double);
        int target_label = (int)target_label_double;

        // Use relation:
        // L = -logit_c + \log \sum_k e^{logit_k}

        // Compute -logit_c
        double logit_target_label = 0;
        tensor2d_get(logits, i, target_label, &logit_target_label);

        // Compute \sum_k e^{logit_k}
        double softmax_normalization = compute_softmax_normalization_f64(logits, i);

        loss_data[0] += (-logit_target_label + log(softmax_normalization));
    }
    loss_data[0] /= batch_size;
}

static void cross_entropy_loss_unchecked_f32(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss)
{
    float batch_size = logits->shape[0];
    float *loss_data = (float *)loss->data;
    loss_data[0] = 0;
    for (size_t i = 0; i < batch_size; i++)
    {
        // Should add support for different data integer
        float target_label_float = 0;
        tensor2d_get(targets, i, 0, &target_label_float);
        int target_label = (int)target_label_float;

        // Use relation:
        // L = -logit_c + \log \sum_k e^{logit_k}

        // Compute -logit_c
        float logit_target_label = 0;
        tensor2d_get(logits, i, target_label, &logit_target_label);

        // Compute \sum_k e^{logit_k}
        float softmax_normalization = compute_softmax_normalization_f32(logits, i);

        loss_data[0] += (-logit_target_label + logf(softmax_normalization));
    }
    loss_data[0] /= batch_size;
}

cgrad_error cross_entropy_loss_graph(struct tensor *const logits, struct tensor *const targets, struct tensor *const loss, struct autograd_allocators *ag_allocators)
{
    cgrad_error err = cross_entropy_loss(logits, targets, loss);
    if (err != NO_ERROR)
    {
        return err;
    }

    // Setup connections
    // In CrossEntropy, targets are not differentiable, so only the logits node is added. Still, the target tensor is added as operand for backward.
    err = add_computational_graph_link(logits, CROSS_ENTROPY_PREDICTED, loss, &cross_entropy_loss_backpropagate_predicted, ag_allocators);
    if (err != NO_ERROR)
    {
        return err;
    }

    // Setup operands manually, as the target was not added to the computational graph as node
    computational_graph_node_set_context_tensor(loss->node, targets, CROSS_ENTROPY_TARGET);

    return NO_ERROR;
}

static void cross_entropy_loss_backpropagate_predicted(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->cgrad_dtype)
    {
    case DTYPE_FLOAT64:
        cross_entropy_loss_backpropagate_predicted_f64(ctx, grad_wrt_out, grad_wrt_operand);
        break;
    case DTYPE_FLOAT32:
        cross_entropy_loss_backpropagate_predicted_f32(ctx, grad_wrt_out, grad_wrt_operand);
        break;
    default:
        break;
    }
}

static void cross_entropy_loss_backpropagate_predicted_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *logits = ctx->operands[CROSS_ENTROPY_PREDICTED];
    const struct tensor *targets = ctx->operands[CROSS_ENTROPY_TARGET];
    double batch_size = logits->shape[0];
    size_t num_classes = logits->shape[1];
    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;

    for (size_t i = 0; i < batch_size; i++)
    {
        double target_label_double = 0;
        tensor2d_get(targets, i, 0, &target_label_double);
        int target_label = (int)target_label_double;

        double softmax_normalization = compute_softmax_normalization_f64(logits, i);

        for (size_t j = 0; j < num_classes; j++)
        {
            double logit = 0;
            tensor2d_get(logits, i, j, &logit);

            double predicted = exp(logit) / softmax_normalization;
            double target = target_label == j ? 1 : 0;

            // dL/dlogit_j = (predicted_j - target_j)
            grad_wrt_operand_data[i * num_classes + j] = (predicted - target) / batch_size;
        }
    }
}

static void cross_entropy_loss_backpropagate_predicted_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *logits = ctx->operands[CROSS_ENTROPY_PREDICTED];
    const struct tensor *targets = ctx->operands[CROSS_ENTROPY_TARGET];
    float batch_size = logits->shape[0];
    size_t num_classes = logits->shape[1];
    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;

    for (size_t i = 0; i < batch_size; i++)
    {
        float target_label_float = 0;
        tensor2d_get(targets, i, 0, &target_label_float);
        int target_label = (int)target_label_float;

        float softmax_normalization = compute_softmax_normalization_f32(logits, i);

        for (size_t j = 0; j < num_classes; j++)
        {
            float logit = 0;
            tensor2d_get(logits, i, j, &logit);

            float predicted = expf(logit) / softmax_normalization;
            float target = target_label == j ? 1 : 0;

            // dL/dlogit_j = (predicted_j - target_j)
            grad_wrt_operand_data[i * num_classes + j] = (predicted - target) / batch_size;
        }
    }
}

static double compute_softmax_normalization_f64(const struct tensor *const logits, const size_t row)
{
    double softmax_normalization = 0;
    size_t logits_size = logits->shape[1];
    for (size_t col = 0; col < logits_size; col++)
    {
        double logit = 0;
        tensor2d_get(logits, row, col, &logit);
        softmax_normalization += exp(logit);
    }

    return softmax_normalization;
}

static float compute_softmax_normalization_f32(const struct tensor *const logits, const size_t row)
{
    float softmax_normalization = 0;
    size_t logits_size = logits->shape[1];
    for (size_t col = 0; col < logits_size; col++)
    {
        float logit = 0;
        tensor2d_get(logits, row, col, &logit);
        softmax_normalization += expf(logit);
    }

    return softmax_normalization;
}