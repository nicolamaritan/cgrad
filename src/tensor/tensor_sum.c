#include "tensor/tensor_sum.h"
#include "string.h"

typedef void (*tensor_sum_reduce)(const struct tensor *const t, const size_t axis, struct tensor *const out, const size_t t_ptr, const size_t out_ptr);

static cgrad_error tensor_sum_dispatch(const struct tensor *const t, const size_t axis, struct tensor *const out);
static void tensor_sum_compute(const struct tensor *const t, const size_t axis, struct tensor *const out, tensor_sum_reduce reduce);
static void tensor_sum_reduce_f64(const struct tensor *const t, const size_t axis, struct tensor *const out, const size_t t_ptr, const size_t out_ptr);

cgrad_error tensor_sum(const struct tensor *const t, const size_t axis, struct tensor *const out)
{
    if (!t || !out)
    {
        return TENSOR_NULL;
    }

    return tensor_sum_dispatch(t, axis, out);
}

static cgrad_error tensor_sum_dispatch(const struct tensor *const t, const size_t axis, struct tensor *const out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        tensor_sum_compute(t, axis, out, &tensor_sum_reduce_f64);
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
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
    double *out_data = out->data;
    double *t_data = t->data;
    for (size_t i = 0; i < t->shape[axis]; i++)
    {
        sum += t_data[t_ptr + i * t->stride[axis]];
    }
    out_data[out_ptr] = sum;
}