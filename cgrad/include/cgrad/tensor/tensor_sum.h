#ifndef TENSOR_SUM_H
#define TENSOR_SUM_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/error.h"

#include <stddef.h>

cgrad_error tensor_sum(const struct tensor *const t, const size_t axis, struct tensor *const out);

#endif