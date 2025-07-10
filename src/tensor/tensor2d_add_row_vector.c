#include "tensor/tensor2d_add_row_vector.h"
#include "autograd/computational_graph.h"
#include "autograd/computational_graph_link.h"
#include <stdlib.h>

typedef enum tensor2d_add_row_vector_operand
{
    TENSOR2D,
    ROW_VECTOR,
} tensor2d_add_row_vector_operand;

static void tensor2d_add_row_vector_backpropagate_tensor2d(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void tensor2d_add_row_vector_backpropagate_row_vector(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor2d_add_row_vector(const struct tensor *const A, const struct tensor *const v, struct tensor *const out)
{
    if (!A || !v)
    {
        return TENSOR_NULL;
    }
    if (!A->data || !v->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (A->shape_size != 2 || v->shape_size != 2)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (v->shape[1] != 1)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (A->shape[1] != v->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    tensor2d_add_row_vector_unchecked(A, v, out);

    return NO_ERROR;
}

cgrad_error tensor2d_add_row_vector_graph(struct tensor *const A, struct tensor *const v, struct tensor *const out, struct autograd_allocators *allocators)
{
    if (!A || !v)
    {
        return TENSOR_NULL;
    }
    if (!A->data || !v->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (A->shape_size != 2 || v->shape_size != 2)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (v->shape[1] != 1)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (A->shape[1] != v->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    tensor2d_add_row_vector_unchecked(A, v, out);

    // Update computational graph
    cgrad_error err = add_computational_graph_link(A, TENSOR2D, out, &tensor2d_add_row_vector_backpropagate_tensor2d, allocators);
    if (err != NO_ERROR) 
    {
        return err;
    }

    err = add_computational_graph_link(v, ROW_VECTOR, out, &tensor2d_add_row_vector_backpropagate_row_vector, allocators);

    return err;
}

void tensor2d_add_row_vector_unchecked(const struct tensor *const A, const struct tensor *const v, struct tensor *out)
{
    size_t rows = A->shape[0];
    size_t cols = A->shape[1];

    double *A_data = A->data;
    double *v_data = v->data;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            // Unchecked is invoked for performance reasons.
            tensor2d_set_unchecked(out, i, j, A_data[i * cols + j] + v_data[j]);
        }
    }
}

static void tensor2d_add_row_vector_backpropagate_tensor2d(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    tensor2d_copy(grad_wrt_out, grad_wrt_operand);
}

static void tensor2d_add_row_vector_backpropagate_row_vector(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    size_t G_rows = grad_wrt_out->shape[0];
    size_t G_cols = grad_wrt_out->shape[1];

    for (size_t j = 0; j < G_cols; j++)
    {
        grad_wrt_operand->data[j] = 0;
    }

    // Iterating by row since vectors are stored in row-major
    for (size_t i = 0; i < G_rows; i++)
    {
        for (size_t j = 0; j < G_cols; j++)
        {
            grad_wrt_operand->data[j] += grad_wrt_out->data[i * G_cols + j];
        }
    }
}