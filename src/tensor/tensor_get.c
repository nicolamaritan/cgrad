#include "tensor/tensor_get.h"
#include "tensor/tensor_helpers.h"

static inline cgrad_error tensor2d_get_check(const struct tensor *t, size_t row, size_t col);

cgrad_error tensor2d_get_f64(const struct tensor *t, size_t row, size_t col, double *out)
{
    if (t->dtype != DTYPE_FLOAT64)
    {
        return TENSOR_INVALID_DTYPE;
    }
    cgrad_error err;
    if ((err = tensor2d_get_check(t, row, col)) != NO_ERROR)
    {
        return err;
    }

    size_t offset = row * t->shape[1] + col;
    (*out) = ((double *)(t->data))[offset];

    return NO_ERROR;
}

cgrad_error tensor2d_get_f32(const struct tensor *t, size_t row, size_t col, float *out)
{
    if (t->dtype != DTYPE_FLOAT32)
    {
        return TENSOR_INVALID_DTYPE;
    }
    cgrad_error err;
    if ((err = tensor2d_get_check(t, row, col)) != NO_ERROR)
    {
        return err;
    }

    size_t offset = row * t->shape[1] + col;
    (*out) = ((float *)(t->data))[offset];

    return NO_ERROR;
}

cgrad_error tensor2d_get_i32(const struct tensor *t, size_t row, size_t col, int32_t *out)
{
    if (t->dtype != DTYPE_INT32)
    {
        return TENSOR_INVALID_DTYPE;
    }
    cgrad_error err;
    if ((err = tensor2d_get_check(t, row, col)) != NO_ERROR)
    {
        return err;
    }

    size_t offset = row * t->shape[1] + col;
    (*out) = ((int32_t *)(t->data))[offset];

    return NO_ERROR;
}

static inline cgrad_error tensor2d_get_check(const struct tensor *t, size_t row, size_t col)
{
    cgrad_error error = tensor_check_null(t);
    if (error != NO_ERROR)
    {
        return error;
    }
    if (t->shape_size != 2)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (row >= t->shape[0] || col >= t->shape[1])
    {
        return TENSOR_INDEX_OUT_OF_BOUNDS;
    }

    return NO_ERROR;
}
