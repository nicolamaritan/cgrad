#ifndef CGRAD_TENSOR_TENSOR_CONST_SCALAR_MULT_H
#define CGRAD_TENSOR_TENSOR_CONST_SCALAR_MULT_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/cgrad_env.h"
#include "cgrad/error.h"
#include <stdbool.h>

cgrad_error tensor_const_scalar_mult(struct tensor *const t, const double scalar, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);

#endif