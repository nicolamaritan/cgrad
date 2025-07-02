#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"

typedef enum cross_entropy_loss_operand
{
    CROSS_ENTROPY_PREDICTED = 0,
    CROSS_ENTROPY_TARGET = 1
} cross_entropy_loss_operand;

cgrad_error cross_entropy_loss(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss);
cgrad_error cross_entropy_loss_graph(struct tensor *const logits, struct tensor *const targets, struct tensor *const loss);
void cross_entropy_loss_backpropagate_predicted(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
void cross_entropy_loss_backpropagate_target(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

#endif