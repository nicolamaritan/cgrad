#include "tensor/tensor2d_trans.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"

typedef enum tensor2d_trans_operand
{
    TENSOR2D_TRANS_ONLY_OPERAND,
} tensor2d_trans_operand;

static cgrad_error tensor2d_trans_dispatch(const struct tensor *const t, struct tensor *const out);
static void tensor2d_trans_unchecked_f64(const struct tensor *const t, struct tensor *const out);
static void tensor2d_trans_unchecked_f32(const struct tensor *const t, struct tensor *const out);
static void tensor2d_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor2d_trans(const struct tensor *const t, struct tensor *const out)
{
    if (!t || !out)
    {
        return TENSOR_NULL;
    }
    if (!t->data || !out->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (t->shape[0] != out->shape[1] || t->shape[1] != out->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    return tensor2d_trans_dispatch(t, out);
}

cgrad_error tensor2d_trans_graph(struct tensor *const t, struct tensor *const out, struct autograd_allocators *allocators)
{
    cgrad_error err = tensor2d_trans(t, out);

    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(t, TENSOR2D_TRANS_ONLY_OPERAND, out, &tensor2d_trans_backpropagate, allocators);
    return err;
}

static cgrad_error tensor2d_trans_dispatch(const struct tensor *const t, struct tensor *const out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        tensor2d_trans_unchecked_f64(t, out);
        break;
    case DTYPE_FLOAT32:
        tensor2d_trans_unchecked_f32(t, out);
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void tensor2d_trans_unchecked_f64(const struct tensor *const t, struct tensor *const out)
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
}

static void tensor2d_trans_unchecked_f32(const struct tensor *const t, struct tensor *const out)
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
}

static void tensor2d_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    tensor2d_trans(grad_wrt_out, grad_wrt_operand);
}