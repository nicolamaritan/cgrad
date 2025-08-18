#ifndef MSE_H
#define MSE_H

#include "cgrad/memory/allocators.h"

cgrad_error mse_loss(struct tensor *const y_pred, struct tensor *const y_target, struct tensor **const z, const bool track_grad, struct allocators *const allocs);

#endif