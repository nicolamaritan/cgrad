#ifndef TENSOR_AXPY_H
#define TENSOR_AXPY_H

#include "tensor/tensor.h"
#include "utils/error.h"

cgrad_error tensor_axpy(struct tensor *const X, struct tensor *const Y, const double a);

#endif