#ifndef TENSOR_SCALAR_MULT_TENSOR_SUM_H
#define TENSOR_SCALAR_MULT_TENSOR_SUM_H

#include "tensor/tensor.h"
#include "utils/error.h"

cgrad_error tensor_scalar_mult_tensor_sum(struct tensor *const A, struct tensor *const B, const double alpha, struct tensor *const out);

#endif