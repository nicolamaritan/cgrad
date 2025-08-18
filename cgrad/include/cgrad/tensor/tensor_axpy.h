#ifndef TENSOR_AXPY_H
#define TENSOR_AXPY_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/error.h"

cgrad_error tensor_axpy(const struct tensor *const x, struct tensor *const y, const double a);

#endif