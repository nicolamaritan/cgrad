#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"

typedef enum cross_entropy_loss_operand {
    CROSS_ENTROPY_PREDICTED = 0,
    CROSS_ENTROPY_TARGET = 1
} cross_entropy_loss_operand;

cgrad_error cross_entropy_loss(const tensor* const logits, const tensor* const targets, tensor* const loss);
cgrad_error cross_entropy_loss_graph(tensor* const logits, tensor* const targets, tensor* const loss);
void cross_entropy_loss_backpropagate_predicted(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);
void cross_entropy_loss_backpropagate_target(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);

#endif