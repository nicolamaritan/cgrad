#include "tensor/tensor2d_trans.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"

typedef enum tensor2d_trans_operand
{
    TENSOR2D_TRANS_ONLY_OPERAND,
} tensor2d_trans_operand;

static cgrad_error tensor2d_trans_dispatch(const struct tensor *const t, struct tensor *const out);
static cgrad_error tensor2d_trans_f64(const struct tensor *const t, struct tensor *const out);
static cgrad_error tensor2d_trans_f32(const struct tensor *const t, struct tensor *const out);
static cgrad_error tensor2d_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor2d_trans(const struct tensor *const t, struct tensor **const out, struct tensor_allocator *const tensor_alloc)
{
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (!t->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (t->shape_size != 2)
    {
        return TENSOR_WRONG_SHAPE;
    }
    
    const size_t shape[] = {t->shape[1], t->shape[0]};
    const size_t shape_size = 2;
    (*out) = tensor_allocator_alloc(tensor_alloc, shape, shape_size, t->dtype);
    if (!(*out))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    return tensor2d_trans_dispatch(t, *out);
}

cgrad_error tensor2d_trans_graph(struct tensor *const t, struct tensor **const out, struct allocators *allocs)
{
    cgrad_error err = tensor2d_trans(t, out, allocs->tensor_alloc);

    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(t, TENSOR2D_TRANS_ONLY_OPERAND, *out, &tensor2d_trans_backpropagate, allocs);
    return err;
}

cgrad_error tensor2d_trans_into(const struct tensor *const t, struct tensor *const out)
{
    const size_t EXPECTED_SHAPE_SIZE = 2;

    if (!t || !out)
    {
        return TENSOR_NULL;
    }
    if (!t->data || !out->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (t->shape_size != EXPECTED_SHAPE_SIZE || out->shape_size != EXPECTED_SHAPE_SIZE)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (t->shape[0] != out->shape[1] || t->shape[1] != out->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    return tensor2d_trans_dispatch(t, out);
}

static cgrad_error tensor2d_trans_dispatch(const struct tensor *const t, struct tensor *const out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor2d_trans_f64(t, out);
    case DTYPE_FLOAT32:
        return tensor2d_trans_f32(t, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor2d_trans_f64(const struct tensor *const t, struct tensor *const out)
{
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    double *restrict out_data = (double *)out->data;
    double *restrict t_data = (double *)t->data;

    // Transpose
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            out_data[j * rows + i] = t_data[i * cols + j];
        }
    }

    return NO_ERROR;
}

static cgrad_error tensor2d_trans_f32(const struct tensor *const t, struct tensor *const out)
{
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    float *restrict out_data = (float *)out->data;
    float *restrict t_data = (float *)t->data;

    // Transpose
    for (size_t i = 0; i < rows; i++)
    {
        size_t offset = i * cols;
        for (size_t j = 0; j < cols; j++)
        {
            out_data[j * rows + i] = t_data[offset + j];
        }
    }

    return NO_ERROR;
}

static cgrad_error tensor2d_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    return tensor2d_trans_into(grad_wrt_out, grad_wrt_operand);
}