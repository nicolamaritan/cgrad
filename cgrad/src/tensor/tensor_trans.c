#include "cgrad/tensor/tensor_trans.h"

#include "cgrad/autograd/backpropagation/backpropagation_context.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"

typedef enum tensor_trans_operand
{
    TENSOR,
} tensor_trans_operand;

typedef enum tensor_trans_operand_size_t
{
    AXIS_1,
    AXIS_2
} tensor_trans_operand_size_t;
    
static inline cgrad_error tensor_trans_update_graph(struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor **const out, struct cgrad_env *env);
static cgrad_error tensor_trans_dispatch(const struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor *const out);
// static cgrad_error tensor_trans_f64(const struct tensor *const t, struct tensor *const out);
static cgrad_error tensor_trans_f32(const struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor *const out);
static cgrad_error tensor_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor_trans(struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor **const out, const bool track_grad, struct cgrad_env *const env)
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

    (*out) = tensor_allocator_alloc(&env->tensor_alloc, trans_shape, t->shape_size, t->dtype);
    if (!(*out))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    cgrad_error err = tensor_trans_dispatch(t, axis_1, axis_2, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor_trans_update_graph(t, axis_1, axis_2, out, env);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_trans_update_graph(struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor **const out, struct cgrad_env *env)
{
    cgrad_error err = add_computational_graph_link(t, TENSOR, *out, &tensor_trans_backpropagate, env);
    if (err != NO_ERROR)
    {
        return err;
    }

    /**
     * Save axeses for backpropagation. They are needed 
     * to perform the inverse operation during backprop, that is
     * transposing the gradient to the original shape.
     */
    err = context_set_operand_size_t(&(*out)->node->ctx, axis_1, AXIS_1);
    if (err != NO_ERROR)
    {
        return err;
    }

    return context_set_operand_size_t(&(*out)->node->ctx, axis_2, AXIS_2);
}

cgrad_error tensor_trans_into(const struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor *const out)
{
    if (!t || !out)
    {
        return TENSOR_NULL;
    }
    if (!t->data || !out->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (t->shape_size != out->shape_size)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (t->shape[axis_1] != out->shape[axis_2] || t->shape[axis_2] != out->shape[axis_1])
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    return tensor_trans_dispatch(t, axis_1, axis_2, out);
}

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

static cgrad_error tensor_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const size_t axis_1 = ctx->operands_size_t[AXIS_1];
    const size_t axis_2 = ctx->operands_size_t[AXIS_2];
    return tensor_trans_into(grad_wrt_out, axis_1, axis_2, grad_wrt_operand);
}