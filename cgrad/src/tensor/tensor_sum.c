#include "cgrad/tensor/tensor_sum.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

typedef void (*tensor_sum_reduce)(const struct tensor *const t, const size_t axis, struct tensor *const out, const size_t t_ptr, const size_t out_ptr);

static cgrad_error tensor_sum_dispatch(const struct tensor *const t, const size_t axis, struct tensor *const out);
static void tensor_sum_compute(const struct tensor *const t, const size_t axis, struct tensor *const out, tensor_sum_reduce reduce);
static void tensor_sum_reduce_f64(const struct tensor *const t, const size_t axis, struct tensor *const out, const size_t t_ptr, const size_t out_ptr);
static void tensor_sum_reduce_f32(const struct tensor *const t, const size_t axis, struct tensor *const out, const size_t t_ptr, const size_t out_ptr);

cgrad_error tensor_sum(const struct tensor *const t, const size_t axis, struct tensor *const out)
{
    if (!t || !out)
    {
        return TENSOR_NULL;
    }
    if (out->shape[axis] != 1)
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    for (size_t i = 0; i < t->shape_size; i++)
    {
        if (i != axis && t->shape[i] != out->shape[i])
        {
            return TENSOR_SHAPE_MISMATCH;
        }
    }

    return tensor_sum_dispatch(t, axis, out);
}

static cgrad_error tensor_sum_dispatch(const struct tensor *const t, const size_t axis, struct tensor *const out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        tensor_sum_compute(t, axis, out, &tensor_sum_reduce_f64);
        break;
    case DTYPE_FLOAT32:
        tensor_sum_compute(t, axis, out, &tensor_sum_reduce_f32);
        break;
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }

    return NO_ERROR;
}

static void tensor_sum_compute(const struct tensor *const t, const size_t axis, struct tensor *const out, tensor_sum_reduce reduce)
{
    for (size_t out_ptr = 0; out_ptr < out->data_size; out_ptr++)
    {
        // Create index
        size_t out_idx[out->shape_size];

        // Compute out_idx unravel
        // TODO This can be optimized by incrementally updating current index
        size_t curr_out_ptr = out_ptr;
        for (size_t i = 0; i < out->shape_size; i++)
        {
            out_idx[i] = curr_out_ptr / out->stride[i];
            curr_out_ptr = curr_out_ptr % out->stride[i];
        }

        // Compute t_ptr, reduction starting point
        size_t t_ptr = 0;
        for (size_t i = 0; i < t->shape_size; i++)
        {
            t_ptr += out_idx[i] * t->stride[i];
        }

        reduce(t, axis, out, t_ptr, out_ptr);
    }
}

static void tensor_sum_reduce_f64(const struct tensor *const t, const size_t axis, struct tensor *const out, const size_t t_ptr, const size_t out_ptr)
{
    double sum = 0;
    double *restrict out_data = out->data;
    double *restrict t_data = t->data;
    for (size_t i = 0; i < t->shape[axis]; i++)
    {
        sum += t_data[t_ptr + i * t->stride[axis]];
    }
    out_data[out_ptr] = sum;
}

static void tensor_sum_reduce_f32(const struct tensor *const t, const size_t axis, struct tensor *const out, const size_t t_ptr, const size_t out_ptr)
{
    float sum = 0;
    float *restrict out_data = out->data;
    float *restrict t_data = t->data;
    for (size_t i = 0; i < t->shape[axis]; i++)
    {
        sum += t_data[t_ptr + i * t->stride[axis]];
        // printf("%ld\n%ld\n\n", t_ptr + i * t->stride[axis], t->data_size);
        assert(t_ptr + i * t->stride[axis] < t->data_size);
    }
    out_data[out_ptr] = sum;
    assert(out_ptr < out->data_size);
}