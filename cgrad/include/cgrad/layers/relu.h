#ifndef RELU_H
#define RELU_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/memory/allocators.h"
#include <stddef.h>

cgrad_error relu_forward(struct tensor *const x, struct tensor **const out, const bool track_grad, struct allocators *const allocs);

#endif