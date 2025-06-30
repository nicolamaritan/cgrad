#ifndef MSE_H
#define MSE_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"

typedef enum mse_loss_operand
{
    MSE_PREDICTED = 0,
    MSE_TARGET = 1
} mse_loss_operand;

cgrad_error mse_loss(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
cgrad_error mse_loss_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor *const out);
void mse_loss_backpropagate_predicted(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
void mse_loss_backpropagate_target(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

#endif