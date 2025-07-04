#ifndef MSE_H
#define MSE_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"

cgrad_error mse_loss(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
cgrad_error mse_loss_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor *const out);

#endif