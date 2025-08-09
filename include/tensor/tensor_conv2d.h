#ifndef TENSOR_CONV2D_H
#define TENSOR_CONV2D_H

#include "memory/allocators.h"

cgrad_error tensor_conv2d(struct tensor *const x, struct tensor *const kernel, struct tensor **const out, const bool track_grad, struct allocators *const allocs);

#endif