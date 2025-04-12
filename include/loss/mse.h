#ifndef MSE_H
#define MSE_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"

typedef enum mse_loss_operand {
    MSE_PREDICTED = 0,
    MSE_TARGET = 1
} mse_loss_operand;

cgrad_error mse_loss(const tensor* const y_pred, const tensor* const y_target, tensor* const z);
cgrad_error mse_loss_graph(tensor* const y_pred, tensor* const y_target, tensor* const out);
void mse_loss_backpropagate_predicted(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);
void mse_loss_backpropagate_target(const tensor **const operands, const tensor* const grad_wrt_out, tensor* grad_wrt_operand);

#endif