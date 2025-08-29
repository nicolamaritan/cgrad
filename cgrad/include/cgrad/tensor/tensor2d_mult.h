#ifndef TENSOR2D_MULT_H
#define TENSOR2D_MULT_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/autograd/backpropagation/backpropagation_function.h"
#include "cgrad/cgrad_env.h"

cgrad_error tensor2d_mult(struct tensor *const lhs, struct tensor *const rhs, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);
cgrad_error tensor2d_mult_into(const struct tensor *const lhs, const struct tensor *const rhs, struct tensor *const out);

#endif