#ifndef TENSOR_AXPY_H
#define TENSOR_AXPY_H

#include "tensor/tensor.h"
#include "utils/error.h"

cgrad_error tensor_axpy(const struct tensor *const x, struct tensor *const y, const double a);

#endif