#ifndef TENSOR_GET_H
#define TENSOR_GET_H

#include "tensor/tensor.h"
#include "tensor/dtypes.h"

#define tensor2d_get(t, row, col, out) \
    _Generic((out), \
    double *: tensor2d_get_f64, \
    float *: tensor2d_get_f32, \
    int32_t *: tensor2d_get_i32)(t, row, col, out)

cgrad_error tensor2d_get_f64(const struct tensor *t, size_t row, size_t col, double *out);
cgrad_error tensor2d_get_f32(const struct tensor *t, size_t row, size_t col, float *out);
cgrad_error tensor2d_get_i32(const struct tensor *t, size_t row, size_t col, int32_t *out);

#endif