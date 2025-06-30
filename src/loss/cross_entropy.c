#include "loss/cross_entropy.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double compute_softmax_normalization(const struct tensor* const logits, const size_t row);

cgrad_error cross_entropy_loss(const struct tensor* const logits, const struct tensor* const targets, struct tensor* const loss)
{
    if (!logits|| !targets|| !loss)
    {
        return TENSOR_NULL;
    }
    if (!logits->data || !targets->data || !loss->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (!logits->shape || !targets->shape || !loss->shape)
    {
        return TENSOR_SHAPE_NULL;
    }
    // TODO add check on tensor shape

    double batch_size = logits->shape[0];
    loss->data[0] = 0;
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
        double softmax_normalization = compute_softmax_normalization(logits, i);

        loss->data[0] += (-logit_target_label + log(softmax_normalization));
    }
    loss->data[0] /= batch_size;

    return NO_ERROR;
}

cgrad_error cross_entropy_loss_graph(struct tensor* const logits, struct tensor* const targets, struct tensor* const loss)
{
    cgrad_error err = cross_entropy_loss(logits, targets, loss);
    if (err != NO_ERROR)
        return err;

    struct computational_graph_node *logits_node = logits->node ? logits->node : computational_graph_node_tensor_alloc(logits);
    struct computational_graph_node *targets_node = targets->node ? targets->node : computational_graph_node_tensor_alloc(targets);

    targets_node->t = (struct tensor *)targets;

    struct computational_graph_node *loss_node = computational_graph_node_tensor_alloc(loss);

    // Setup connections
    // In CrossEntropy, targets are not differentiable, so only the logits node is added. Still, the target tensor is added as operand for backward.
    add_parent(logits_node, loss_node, CROSS_ENTROPY_PREDICTED);
    add_child(loss_node, logits_node);

    // Setup backpropation functions 
    loss_node->function[CROSS_ENTROPY_PREDICTED] = (backpropagation_function)&cross_entropy_loss_backpropagate_predicted;

    // Setup operands
    loss_node->tensor_operands[CROSS_ENTROPY_PREDICTED] = logits;
    loss_node->tensor_operands[CROSS_ENTROPY_TARGET] = targets;

    return NO_ERROR;
}

void cross_entropy_loss_backpropagate_predicted(const struct tensor **const operands, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    const struct tensor *logits = operands[CROSS_ENTROPY_PREDICTED];
    const struct tensor *targets= operands[CROSS_ENTROPY_TARGET];
    double batch_size = logits->shape[0];
    size_t num_classes = logits->shape[1];

    for (size_t i = 0; i < batch_size; i++)
    {
        double target_label_double = 0;
        tensor2d_get(targets, i, 0, &target_label_double);
        int target_label = (int)target_label_double;

        double softmax_normalization = compute_softmax_normalization(logits, i);

        for (size_t j = 0; j < num_classes; j++)
        {
            double logit = 0;
            tensor2d_get(logits, i, j, &logit);

            double predicted = exp(logit) / softmax_normalization;
            double target = target_label == j ? 1 : 0;

            // dL/dlogit_j = (predicted_j - target_j)
            grad_wrt_operand->data[i * num_classes + j] = (predicted - target) / batch_size;
        }
    }
}

double compute_softmax_normalization(const struct tensor* const logits, const size_t row)
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