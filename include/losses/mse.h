#ifndef MSE_H
#define MSE_H

#include "autograd/autograd_allocators.h"

cgrad_error mse_loss(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor *const z);
cgrad_error mse_loss_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor *const out, struct autograd_allocators *ag_allocators);

#endif