#include "cgrad/tensor/tensor_add.h"
#include "cgrad/tensor/tensor_copy.h"
#include "cgrad/tensor/tensor_helpers.h"
#include "cgrad/tensor/tensor_equality.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"

typedef enum tensor_add_operand
{
    LHS_TENSOR,
    RHS_TENSOR,
} tensor_add_operand;

static inline cgrad_error tensor_add_update_graph(struct tensor *const x, struct tensor *const y, struct tensor **const out, struct allocators *const allocs);
static inline cgrad_error tensor_add_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor_add_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor_add_f32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);
static cgrad_error tensor_add_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor_add(struct tensor *const x, struct tensor *const y, struct tensor **const out, const bool track_grad, struct allocators *const allocs)
{
    if (!allocs)
    {
        return ALLOCATORS_NULL;
    }
    if (!x || !y)
    {
        return TENSOR_NULL;
    }
    if (!x->data || !y->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (x->data_size != y->data_size)
    {
        return TENSOR_DATA_SIZE_MISMATCH;
    }
    if (!tensor_same_shape(x, y))
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (x->dtype != y->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    (*out) = tensor_allocator_alloc(allocs->tensor_alloc, x->shape, x->shape_size, x->dtype);

    cgrad_error err = tensor_add_dispatch(x, y, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor_add_update_graph(x, y, out, allocs);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_add_update_graph(struct tensor *const x, struct tensor *const y, struct tensor **const out, struct allocators *const allocs)
{
    cgrad_error err = add_computational_graph_link(x, LHS_TENSOR, *out, &tensor_add_backpropagate, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(y, RHS_TENSOR, *out, &tensor_add_backpropagate, allocs);

    return err;
}

static cgrad_error tensor_add_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    /**
     * Given the symmetry of the addition operation, the gradient with respect to both operands is the same.
     * Therefore, we can use the same gradient for both operands.
     * The gradient with respect to both operands is the gradient with respect to the output.
     */
    return tensor_copy(grad_wrt_out, grad_wrt_operand);
}

static inline cgrad_error tensor_add_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    switch (x->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor_add_f64(x, y, out);
    case DTYPE_FLOAT32:
        return tensor_add_f32(x, y, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_add_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    double *restrict out_data = (double *)out->data;
    double *restrict A_data = (double *)x->data;
    double *restrict B_data = (double *)y->data;

    for (size_t i = 0; i < x->data_size; i++)
    {
        out_data[i] = A_data[i] + B_data[i];
    }

    return NO_ERROR;
}

static cgrad_error tensor_add_f32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    float *restrict out_data = (float *)out->data;
    float *restrict A_data = (float *)x->data;
    float *restrict B_data = (float *)y->data;

    for (size_t i = 0; i < x->data_size; i++)
    {
        out_data[i] = A_data[i] + B_data[i];
    }

    return NO_ERROR;
}
