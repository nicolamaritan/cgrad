#include "tensor/tensor2d_trans.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"

typedef enum tensor2d_trans_operand
{
    TENSOR2D_TRANS_ONLY_OPERAND,
} tensor2d_trans_operand;

static void tensor2d_trans_unchecked(const struct tensor *const t, struct tensor *const out);
static void tensor2d_trans_unchecked_f64(const struct tensor *const t, struct tensor *const out);
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

    tensor2d_trans_unchecked(t, out);
    return NO_ERROR;
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

static void tensor2d_trans_unchecked(const struct tensor *const t, struct tensor *const out)
{
    switch (t->dtype)
    {
        case DTYPE_FLOAT64:
            tensor2d_trans_unchecked_f64(t, out);
            break;
        default:
            break;
    }
}

static void tensor2d_trans_unchecked_f64(const struct tensor *const t, struct tensor *const out)
{
    // Extract the shape of t 
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    double *out_data = (double *)out->data;
    double *t_data = (double *)t->data;

    out->shape[0] = cols;
    out->shape[1] = rows;

    // Transpose
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            out_data[j * rows + i] = t_data[i * cols + j];
        }
    }
}

static void tensor2d_trans_backpropagate(const struct backpropagation_context *const ctx, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    tensor2d_trans(grad_wrt_out, grad_wrt_operand);
}