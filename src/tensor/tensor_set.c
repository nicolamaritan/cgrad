#include "tensor/tensor_set.h"

static inline cgrad_error tensor2d_set_check(struct tensor *t, size_t row, size_t col);

cgrad_error tensor2d_set_f64(struct tensor *t, size_t row, size_t col, double value)
{
    if (t->dtype != DTYPE_FLOAT64)
    {
        return TENSOR_INVALID_DTYPE;
    }
    cgrad_error err;
    if ((err = tensor2d_set_check(t, row, col)) != NO_ERROR)
    {
        return err;
    }

    size_t offset = row * t->shape[1] + col;
    ((double *)t->data)[offset] = value;

    return NO_ERROR;
}

cgrad_error tensor2d_set_f32(struct tensor *t, size_t row, size_t col, float value)
{
    if (t->dtype != DTYPE_FLOAT32)
    {
        return TENSOR_INVALID_DTYPE;
    }
    cgrad_error err;
    if ((err = tensor2d_set_check(t, row, col)) != NO_ERROR)
    {
        return err;
    }

    size_t offset = row * t->shape[1] + col;
    ((float *)t->data)[offset] = value;

    return NO_ERROR;
}

cgrad_error tensor2d_set_i32(struct tensor *t, size_t row, size_t col, int32_t value)
{
    if (t->dtype != DTYPE_INT32)
    {
        return TENSOR_INVALID_DTYPE;
    }
    cgrad_error err;
    if ((err = tensor2d_set_check(t, row, col)) != NO_ERROR)
    {
        return err;
    }

    size_t offset = row * t->shape[1] + col;
    ((int32_t *)t->data)[offset] = value;

    return NO_ERROR;
}

static inline cgrad_error tensor2d_set_check(struct tensor *t, size_t row, size_t col)
{
    if (t == NULL)
    {
        return TENSOR_NULL;
    }
    if (t->data == NULL)
    {
        return TENSOR_DATA_NULL;
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
