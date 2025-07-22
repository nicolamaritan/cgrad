#ifndef TENSOR_SET_H
#define TENSOR_SET_H

#include "tensor/tensor.h"
#include "dtypes.h"

#include <stddef.h>

#define tensor2d_set(t, row, col, value) \
    _Generic((value), \
    double: tensor2d_set_f64, \
    float: tensor2d_set_f32, \
    int32_t: tensor2d_set_i32 \
    )(t, row, col, value)

cgrad_error tensor2d_set_f64(struct tensor *t, size_t row, size_t col, double value);
cgrad_error tensor2d_set_f32(struct tensor *t, size_t row, size_t col, float value);
cgrad_error tensor2d_set_i32(struct tensor *t, size_t row, size_t col, int32_t value);

#endif