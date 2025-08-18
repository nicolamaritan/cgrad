#ifndef TENSOR_ADD_INPLACE_H
#define TENSOR_ADD_INPLACE_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/error.h"

cgrad_error tensor_add_inplace(struct tensor *A, const struct tensor *const B);

#endif