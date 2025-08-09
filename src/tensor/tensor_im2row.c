#include "tensor/tensor_im2row.h"

static inline cgrad_error tensor_im2row_dispatch(struct tensor *t, const struct tensor *kernel, struct tensor **out, struct allocators *allocs);
static cgrad_error tensor_im2row_f32(struct tensor *t, const struct tensor *kernel, struct tensor **out, struct allocators *allocs);

cgrad_error tensor_im2row(struct tensor *t, const struct tensor *kernel, struct tensor **out, const bool track_grad, struct allocators *allocs)
{
    // checks

    tensor_im2row_dispatch(t, kernel, out, allocs);

    return NO_ERROR;
}

static inline cgrad_error tensor_im2row_dispatch(struct tensor *t, const struct tensor *kernel, struct tensor **out, struct allocators *allocs)
{
    switch (t->dtype)
    {
        case DTYPE_FLOAT32:
            return tensor_im2row_f32(t, kernel, out, allocs);
        default:
            return 70;
    }
}

static cgrad_error tensor_im2row_f32(struct tensor *t, const struct tensor *kernel, struct tensor **out, struct allocators *allocs)
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
    float *out_data = (float *)(*out)->data;

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