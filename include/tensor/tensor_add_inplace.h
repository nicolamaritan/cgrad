#ifndef TENSOR_ADD_INPLACE_H
#define TENSOR_ADD_INPLACE_H

#include "tensor/tensor.h"
#include "error.h"

cgrad_error tensor_add_inplace(struct tensor *A, const struct tensor *const B);

#endif