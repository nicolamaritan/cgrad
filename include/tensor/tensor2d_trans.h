#ifndef TENSOR2D_TRANS_H
#define TENSOR2D_TRANS_H

#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation_function.h"
#include "memory/allocators.h"

cgrad_error tensor2d_trans(struct tensor *const t, struct tensor **const out, const bool track_grad, struct allocators *const allocs);
cgrad_error tensor2d_trans_into(const struct tensor *const t, struct tensor *const out);

#endif