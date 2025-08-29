#ifndef TENSOR2D_TRANS_H
#define TENSOR2D_TRANS_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/autograd/backpropagation/backpropagation_function.h"
#include "cgrad/cgrad_env.h"

cgrad_error tensor2d_trans(struct tensor *const t, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);
cgrad_error tensor2d_trans_into(const struct tensor *const t, struct tensor *const out);

#endif