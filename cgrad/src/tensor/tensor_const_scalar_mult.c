#include "cgrad/tensor/tensor_const_scalar_mult.h"
#include "cgrad/tensor/tensor_helpers.h"
#include "cgrad/autograd/backpropagation/backpropagation_context.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include <stddef.h>

typedef enum tensor_const_scalar_mult_operand
{
    TENSOR,
} tensor_const_scalar_mult_operand;

typedef enum tensor_const_scalar_mult_owned
{
    SCALAR,
} tensor_const_scalar_mult_owned;

static inline cgrad_error tensor_const_scalar_mult_update_graph(struct tensor *const t, struct tensor *const out, const double scalar, struct cgrad_env *const env);

static inline cgrad_error tensor_const_scalar_mult_dispatch(const struct tensor *const t, const double scalar, struct tensor *const out);

static cgrad_error tensor_const_scalar_mult_f64(const struct tensor *const t, const double scalar, struct tensor *const out);

static cgrad_error tensor_const_scalar_mult_f32(const struct tensor *const t, const double scalar, struct tensor *const out);

static cgrad_error tensor_const_scalar_mult_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

static cgrad_error tensor_const_scalar_mult_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

static cgrad_error tensor_const_scalar_mult_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

// Helper: allocates and fills the [1]-shaped tensor owned by the output
// node, holding the constant scalar value for later use in the backward
// pass. It is never added to the graph as an operand.
static inline cgrad_error tensor_const_scalar_mult_alloc_owned_scalar(const double scalar, const cgrad_dtype dtype, struct tensor **const scalar_tensor, struct cgrad_env *const env);

cgrad_error tensor_const_scalar_mult(struct tensor *const t, const double scalar, struct tensor **const out, const bool track_grad, struct cgrad_env *const env)
{
    if (!env)
    {
        return ALLOCATORS_NULL;
    }
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (!t->data)
    {
        return TENSOR_DATA_NULL;
    }

    (*out) = tensor_allocator_alloc(&env->tensor_alloc, t->shape, t->shape_size, t->dtype);
    if (!(*out))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    cgrad_error err = tensor_const_scalar_mult_dispatch(t, scalar, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor_const_scalar_mult_update_graph(t, *out, scalar, env);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_const_scalar_mult_update_graph(struct tensor *const t, struct tensor *const out, const double scalar, struct cgrad_env *const env)
{
    // t is the only differentiable operand: only it gets a graph edge.
    cgrad_error err = add_computational_graph_link(t, TENSOR, out, &tensor_const_scalar_mult_backpropagate, env);
    if (err != NO_ERROR)
    {
        return err;
    }

    // scalar is a constant, not a graph node: it is stored as a tensor
    // owned by the output node purely so the backward pass can read it back
    struct tensor *scalar_tensor = NULL;
    err = tensor_const_scalar_mult_alloc_owned_scalar(scalar, t->dtype, &scalar_tensor, env);
    if (err != NO_ERROR)
    {
        return err;
    }

    return context_set_owned(&out->node->ctx, scalar_tensor, SCALAR);
}

static inline cgrad_error tensor_const_scalar_mult_alloc_owned_scalar(const double scalar, const cgrad_dtype dtype, struct tensor **const scalar_tensor, struct cgrad_env *const env)
{
    const size_t scalar_shape[] = {1};

    (*scalar_tensor) = tensor_allocator_alloc(&env->tensor_alloc, scalar_shape, 1, dtype);
    if (!(*scalar_tensor))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    switch (dtype)
    {
    case DTYPE_FLOAT64:
        ((double *)(*scalar_tensor)->data)[0] = scalar;
        return NO_ERROR;
    case DTYPE_FLOAT32:
        ((float *)(*scalar_tensor)->data)[0] = (float)scalar;
        return NO_ERROR;
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static inline cgrad_error tensor_const_scalar_mult_dispatch(const struct tensor *const t, const double scalar, struct tensor *const out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor_const_scalar_mult_f64(t, scalar, out);
    case DTYPE_FLOAT32:
        return tensor_const_scalar_mult_f32(t, scalar, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_const_scalar_mult_f64(const struct tensor *const t, const double scalar, struct tensor *const out)
{
    double *restrict out_data = (double *)out->data;
    double *restrict t_data = (double *)t->data;

    for (size_t i = 0; i < t->data_size; i++)
    {
        out_data[i] = t_data[i] * scalar;
    }

    return NO_ERROR;
}

static cgrad_error tensor_const_scalar_mult_f32(const struct tensor *const t, const double scalar, struct tensor *const out)
{
    float *restrict out_data = (float *)out->data;
    float *restrict t_data = (float *)t->data;
    float scalar_f32 = (float)scalar;

    for (size_t i = 0; i < t->data_size; i++)
    {
        out_data[i] = t_data[i] * scalar_f32;
    }

    return NO_ERROR;
}

static cgrad_error tensor_const_scalar_mult_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor_const_scalar_mult_backpropagate_f64(ctx, grad_wrt_out, grad_wrt_operand);
    case DTYPE_FLOAT32:
        return tensor_const_scalar_mult_backpropagate_f32(ctx, grad_wrt_out, grad_wrt_operand);
    default:
        return AUTOGRAD_BACKPROPAGATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_const_scalar_mult_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *scalar_tensor = ctx->owned[SCALAR];
    double scalar = ((double *)scalar_tensor->data)[0];

    double *restrict grad_out_data = (double *)grad_wrt_out->data;
    double *restrict grad_in_data = (double *)grad_wrt_operand->data;

    // dL/dt_i = grad_out_i * scalar
    for (size_t i = 0; i < grad_wrt_operand->data_size; i++)
    {
        grad_in_data[i] = grad_out_data[i] * scalar;
    }

    return NO_ERROR;
}

static cgrad_error tensor_const_scalar_mult_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *scalar_tensor = ctx->owned[SCALAR];
    float scalar = ((float *)scalar_tensor->data)[0];

    float *restrict grad_out_data = (float *)grad_wrt_out->data;
    float *restrict grad_in_data = (float *)grad_wrt_operand->data;

    for (size_t i = 0; i < grad_wrt_operand->data_size; i++)
    {
        grad_in_data[i] = grad_out_data[i] * scalar;
    }

    return NO_ERROR;
}