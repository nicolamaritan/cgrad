#ifndef MSE_H
#define MSE_H

#include "memory/allocators.h"

cgrad_error mse_loss(const struct tensor *const y_pred, const struct tensor *const y_target, struct tensor **const z, struct tensor_allocator *const tensor_allocator);
cgrad_error mse_loss_graph(struct tensor *const y_pred, struct tensor *const y_target, struct tensor **const z, struct allocators *const allocs);

#endif