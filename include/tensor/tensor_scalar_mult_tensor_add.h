#ifndef TENSOR_SCALAR_MULT_TENSOR_ADD_H
#define TENSOR_SCALAR_MULT_TENSOR_ADD_H

#include "tensor/tensor.h"
#include "error.h"

cgrad_error tensor_scalar_mult_tensor_add(struct tensor *const x, struct tensor *const y, const double alpha, struct tensor *const out);

#endif