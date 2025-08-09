#include "tensor/tensor_trans.h"

#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"

typedef enum tensor2d_trans_operand
{
    TENSOR2D_TRANS_ONLY_OPERAND,
} tensor2d_trans_operand;

// static inline cgrad_error tensor_trans_update_graph(struct tensor *const t, struct tensor **const out, struct allocators *const allocs);
static cgrad_error tensor_trans_dispatch(const struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor *const out);
// static cgrad_error tensor_trans_f64(const struct tensor *const t, struct tensor *const out);
static cgrad_error tensor_trans_f32(const struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor *const out);
// static cgrad_error tensor_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor_trans(struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor **const out, const bool track_grad, struct allocators *const allocs)
{
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (!t->data)
    {
        return TENSOR_DATA_NULL;
    }
    
    size_t trans_shape[TENSOR_MAX_SHAPE_SIZE];
    memcpy(trans_shape, t->shape, sizeof(size_t) * t->shape_size);
    size_t temp = trans_shape[axis_1];
    trans_shape[axis_1] = trans_shape[axis_2];
    trans_shape[axis_2] = temp;

    (*out) = tensor_allocator_alloc(allocs->tensor_alloc, trans_shape, t->shape_size, t->dtype);
    if (!(*out))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    cgrad_error err = tensor_trans_dispatch(t, axis_1, axis_2, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    // if (track_grad)
    // {
    //     return tensor_trans_update_graph(t, out, allocs);
    // }

    return NO_ERROR;
}

// static inline cgrad_error tensor_trans_update_graph(struct tensor *const t, struct tensor **const out, struct allocators *allocs)
// {
//     return add_computational_graph_link(t, TENSOR2D_TRANS_ONLY_OPERAND, *out, &tensor2d_trans_backpropagate, allocs);
// }

// cgrad_error tensor_trans_into(const struct tensor *const t, struct tensor *const out)
// {
//     const size_t EXPECTED_SHAPE_SIZE = 2;

//     if (!t || !out)
//     {
//         return TENSOR_NULL;
//     }
//     if (!t->data || !out->data)
//     {
//         return TENSOR_DATA_NULL;
//     }
//     if (t->shape_size != EXPECTED_SHAPE_SIZE || out->shape_size != EXPECTED_SHAPE_SIZE)
//     {
//         return TENSOR_WRONG_SHAPE;
//     }
//     if (t->shape[0] != out->shape[1] || t->shape[1] != out->shape[0])
//     {
//         return TENSOR_SHAPE_MISMATCH;
//     }

//     return tensor_trans_dispatch(t, out);
// }

static cgrad_error tensor_trans_dispatch(const struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor *const out)
{
    switch (t->dtype)
    {
    // case DTYPE_FLOAT64:
    //     return tensor_trans_f64(t, out);
    case DTYPE_FLOAT32:
        return tensor_trans_f32(t, axis_1, axis_2, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

// static cgrad_error tensor_trans_f64(const struct tensor *const t, struct tensor *const out)
// {
//     size_t rows = t->shape[0];
//     size_t cols = t->shape[1];

//     double *restrict out_data = (double *)out->data;
//     double *restrict t_data = (double *)t->data;

//     // Transpose
//     for (size_t i = 0; i < rows; i++)
//     {
//         for (size_t j = 0; j < cols; j++)
//         {
//             out_data[j * rows + i] = t_data[i * cols + j];
//         }
//     }

//     return NO_ERROR;
// }

static cgrad_error tensor_trans_f32(const struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor *const out)
{
    float *restrict out_data = (float *)out->data;
    float *restrict t_data = (float *)t->data;

    size_t idx[TENSOR_MAX_SHAPE_SIZE];
    memset(idx, 0, sizeof(idx));

    for (size_t d = 0; d < t->data_size; d++)
    {
        size_t t_offset = 0;
        size_t out_offset = 0;

        for (size_t i = 0; i < t->shape_size; i++)
        {
            t_offset += idx[i] * t->stride[i];

            size_t out_idx_i = idx[i];
            if (i == axis_1)
            {
                out_idx_i = idx[axis_2];
            }
            else if (i == axis_2)
            {
                out_idx_i = idx[axis_1];
            }

            out_offset += out_idx_i * out->stride[i];
        }

        out_data[out_offset] = t_data[t_offset];

        // Increment idx
        for (size_t i = t->shape_size; i-- > 0; )
        {
            if (++idx[i] < t->shape[i])
            {
                break;
            }
            idx[i] = 0;
        }
    }

    return NO_ERROR;
}

// static cgrad_error tensor_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
// {
//     return tensor2d_trans_into(grad_wrt_out, grad_wrt_operand);
// }