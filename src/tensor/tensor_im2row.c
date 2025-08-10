#include "tensor/tensor_im2row.h"
#include "autograd/backpropagation/backpropagation_context.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"

typedef enum tensor_im2row_operand
{
    TENSOR,
} tensor_im2row_operand;

typedef enum tensor_im2row_owned
{
    ORIGIN_IDXS,
} tensor_im2row_owned;

static inline cgrad_error tensor_im2row_update_graph(struct tensor *const t, struct tensor *const out, struct tensor *const origin_idxs, struct allocators *allocs);
static inline cgrad_error tensor_im2row_dispatch(struct tensor *t, const struct tensor *kernel, struct tensor **const out, struct tensor **const origin_idxs, struct allocators *const allocs);
static cgrad_error tensor_im2row_f32(struct tensor *t, const struct tensor *kernel, struct tensor **const out, struct tensor **const origin_idxs, struct allocators *const allocs);
static cgrad_error tensor_im2row_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error tensor_im2row_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor_im2row(struct tensor *t, const struct tensor *kernel, struct tensor **out, const bool track_grad, struct allocators *const allocs)
{
    // checks

    struct tensor *origin_idxs = NULL;
    tensor_im2row_dispatch(t, kernel, out, &origin_idxs, allocs);

    if (track_grad)
    {
        return tensor_im2row_update_graph(t, *out, origin_idxs, allocs);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_im2row_update_graph(struct tensor *const t, struct tensor *const out, struct tensor *const origin_idxs, struct allocators *allocs)
{
    cgrad_error err = add_computational_graph_link(t, TENSOR, out, &tensor_im2row_backpropagate, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    return context_set_owned(&out->node->ctx, origin_idxs, ORIGIN_IDXS);
}

static inline cgrad_error tensor_im2row_dispatch(struct tensor *t, const struct tensor *kernel, struct tensor **const out, struct tensor **const origin_idxs, struct allocators *const allocs)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT32:
        return tensor_im2row_f32(t, kernel, out, origin_idxs, allocs);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_im2row_f32(struct tensor *t, const struct tensor *kernel, struct tensor **const out, struct tensor **const origin_idxs, struct allocators *const allocs)
{
    float *t_data = (float *)t->data;

    const size_t H_out = t->shape[2] - kernel->shape[2] + 1;
    const size_t W_out = t->shape[3] - kernel->shape[3] + 1;

    // size_t K = kernel->shape[0];
    size_t C = kernel->shape[1];
    size_t R = kernel->shape[2];
    size_t S = kernel->shape[3];

    const size_t out_shape[] = {H_out * W_out * t->shape[0], C * R * S};
    (*out) = tensor_allocator_alloc(allocs->tensor_alloc, out_shape, 2, t->dtype);
    (*origin_idxs) = tensor_allocator_alloc(allocs->tensor_alloc, out_shape, 2, t->dtype);
    float *out_data = (float *)(*out)->data;
    float *origin_idxs_data = (float *)(*origin_idxs)->data;

    const size_t BATCH_OFFSET = C * R * S * H_out * W_out;

    for (size_t batch = 0; batch < t->shape[0]; batch++)
    {
        size_t row = 0;
        for (size_t h_out = 0; h_out < H_out; h_out++)
        {
            for (size_t w_out = 0; w_out < W_out; w_out++)
            {
                size_t col = 0;
                for (size_t c = 0; c < C; c++)
                {
                    for (size_t r = 0; r < R; r++)
                    {
                        for (size_t s = 0; s < S; s++)
                        {
                            size_t h_in = h_out + r;
                            size_t w_in = w_out + s;

                            out_data[col + row * out_shape[1] + batch * BATCH_OFFSET] = t_data[batch * t->stride[0] + c * t->stride[1] + h_in * t->stride[2] + w_in];
                            origin_idxs_data[col + row * out_shape[1] + batch * BATCH_OFFSET] = batch * t->stride[0] + c * t->stride[1] + h_in * t->stride[2] + w_in;

                            col++;
                        }
                    }
                }
                row++;
            }
        }
    }

    return NO_ERROR;
}

static cgrad_error tensor_im2row_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
    case DTYPE_FLOAT32:
        return tensor_im2row_backpropagate_f32(ctx, grad_wrt_out, grad_wrt_operand);
    default:
        return AUTOGRAD_BACKPROPAGATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_im2row_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    struct tensor *origin_idxs = ctx->owned[ORIGIN_IDXS];

    float *grad_wrt_out_data = (float *)grad_wrt_out->data;
    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    float *origin_idxs_data = (float *)origin_idxs->data;

    for (size_t i = 0; i < origin_idxs->data_size; i++)
    {
        grad_wrt_operand_data[(size_t)origin_idxs_data[i]] += grad_wrt_out_data[i];
    }

    return NO_ERROR;
}