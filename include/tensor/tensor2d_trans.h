#ifndef TENSOR2D_TRANS_H
#define TENSOR2D_TRANS_H

#include "tensor/tensor.h"
#include "autograd/backpropagation_function.h"
#include "autograd/autograd_allocators.h"

cgrad_error tensor2d_trans_graph(struct tensor *const t, struct tensor *const out, struct autograd_allocators *allocators);
cgrad_error tensor2d_trans(const struct tensor *const t, struct tensor *const out);
void tensor2d_trans_unchecked(const struct tensor *const t, struct tensor *const out);

#endif