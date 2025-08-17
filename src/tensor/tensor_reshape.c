#include "tensor/tensor_reshape.h"
#include "autograd/backpropagation/backpropagation_context.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"

typedef enum tensor_reshape_operand
{
    TENSOR,
} tensor_reshape_operand;

typedef enum tensor_reshape_operand_size_t
{
    OLD_SHAPE_SIZE,
    OLD_SHAPE_START_POS, 
} tensor_reshape_operand_size_t;
    
static inline cgrad_error tensor_reshape_update_graph(struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor *const out, struct allocators *const allocs);
static inline cgrad_error tensor_reshape_dispatch(const struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor *const out);
static cgrad_error tensor_reshape_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor_reshape(struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor **const out, const bool track_grad, struct allocators *const allocs)
{
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (!t->data)
    {
        return TENSOR_DATA_NULL;
    }

    // Reshaped data size must be the same as original data size
    size_t reshaped_data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        reshaped_data_size *= shape[i];
    }
    if (reshaped_data_size != t->data_size)
    {
        return TENSOR_RESHAPE_INVALID_SHAPE;
    }
    
    (*out) = tensor_allocator_alloc(allocs->tensor_alloc, shape, shape_size, t->dtype);
    if (!(*out))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    cgrad_error err = tensor_reshape_dispatch(t, shape, shape_size, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor_reshape_update_graph(t, shape, shape_size, *out, allocs);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_reshape_dispatch(const struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor *const out)
{
    memcpy(out->data, t->data, t->data_size * dtype_sizeof(t->dtype));
    return NO_ERROR;
}

static inline cgrad_error tensor_reshape_update_graph(struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor *const out, struct allocators *const allocs)
{
    cgrad_error err = add_computational_graph_link(t, TENSOR, out, &tensor_reshape_backpropagate, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    /**
     * Save operands for backpropagation. The original shape is needed
     * to perform the inverse operation during backprop, that is
     * reshaping the gradient to the original shape.
     */
    err = context_set_operand_size_t(&out->node->ctx, t->shape_size, OLD_SHAPE_SIZE);
    if (err != NO_ERROR)
    {
        return err;
    }

    for (size_t i = 0; i < t->shape_size; i++)
    {
        // Save contiguously after OLD_SHAPE_START_POS
        err = context_set_operand_size_t(&out->node->ctx, t->shape[i], OLD_SHAPE_START_POS + i);
        if (err != NO_ERROR)
        {
            return err;
        }
    }

    return NO_ERROR;
}

cgrad_error tensor_reshape_into(const struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor *const out)
{
    if (!t || !out)
    {
        return TENSOR_NULL;
    }
    if (!t->data || !out->data)
    {
        return TENSOR_DATA_NULL;
    }

    return tensor_reshape_dispatch(t, shape, shape_size, out);
}

static cgrad_error tensor_reshape_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const size_t shape_size = ctx->operands_size_t[OLD_SHAPE_SIZE];
    return tensor_reshape_into(grad_wrt_out, &ctx->operands_size_t[OLD_SHAPE_START_POS], shape_size, grad_wrt_operand);
}