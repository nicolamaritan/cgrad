#include "cgrad/layers/gelu.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include <math.h>
#include <stdlib.h>

typedef enum gelu_layer_operand
{
    GELU_ONLY_OPERAND,
} gelu_layer_operand;

static inline cgrad_error gelu_forward_update_graph(struct tensor *const x, struct tensor **const out, struct cgrad_env *const env);
static cgrad_error gelu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error gelu_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error gelu_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error gelu_forward_dispatch(const struct tensor *const x, struct tensor *const out);
static cgrad_error gelu_forward_scalar(const struct tensor *const x, struct tensor *const out);
static cgrad_error gelu_forward_scalar_f64(const struct tensor *const x, struct tensor *const out);
static cgrad_error gelu_forward_scalar_f32(const struct tensor *const x, struct tensor *const out);

static const double GELU_INV_SQRT_2 = 0.70710678118654752440;
static const double GELU_INV_SQRT_2PI = 0.39894228040143267794;

cgrad_error gelu_forward(struct tensor *const x, struct tensor **const out, const bool track_grad, struct cgrad_env *const env)
{
    if (!x)
    {
        return TENSOR_NULL;
    }

    if (!x->data)
    {
        return TENSOR_DATA_NULL;
    }

    (*out) = tensor_allocator_alloc(&env->tensor_alloc, x->shape, x->shape_size, x->dtype);

    cgrad_error err = gelu_forward_dispatch(x, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return gelu_forward_update_graph(x, out, env);
    }

    return NO_ERROR;
}

static inline cgrad_error gelu_forward_update_graph(struct tensor *const x, struct tensor **const out, struct cgrad_env *const env)
{
    return add_computational_graph_link(x, GELU_ONLY_OPERAND, *out, &gelu_backpropagate, env);
}

static cgrad_error gelu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
    case DTYPE_FLOAT64:
        return gelu_backpropagate_f64(ctx, grad_wrt_out, grad_wrt_operand);

    case DTYPE_FLOAT32:
        return gelu_backpropagate_f32(ctx, grad_wrt_out, grad_wrt_operand);

    default:
        return AUTOGRAD_BACKPROPAGATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error gelu_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *const x = ctx->operands[GELU_ONLY_OPERAND];

    if (!x)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_OPERAND_NULL;
    }

    const double *x_data = (const double *)x->data;
    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;
    const double *grad_wrt_out_data = (const double *)grad_wrt_out->data;

    for (size_t i = 0; i < grad_wrt_operand->data_size; i++)
    {
        const double x_value = x_data[i];
        const double erf_value = erf(x_value * GELU_INV_SQRT_2);
        const double gaussian_term = GELU_INV_SQRT_2PI * exp(-0.5 * x_value * x_value);

        const double derivative =
            0.5 * (1.0 + erf_value)
            + x_value * gaussian_term;

        grad_wrt_operand_data[i] = derivative * grad_wrt_out_data[i];
    }

    return NO_ERROR;
}

static cgrad_error gelu_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *const x = ctx->operands[GELU_ONLY_OPERAND];

    if (!x)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_OPERAND_NULL;
    }

    const float *x_data = (const float *)x->data;
    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    const float *grad_wrt_out_data = (const float *)grad_wrt_out->data;

    const float inv_sqrt_2 = (float)GELU_INV_SQRT_2;
    const float inv_sqrt_2pi = (float)GELU_INV_SQRT_2PI;

    for (size_t i = 0; i < grad_wrt_operand->data_size; i++)
    {
        const float x_value = x_data[i];
        const float erf_value = erff(x_value * inv_sqrt_2);
        const float gaussian_term = inv_sqrt_2pi * expf(-0.5f * x_value * x_value);

        const float derivative =
            0.5f * (1.0f + erf_value)
            + x_value * gaussian_term;

        grad_wrt_operand_data[i] = derivative * grad_wrt_out_data[i];
    }

    return NO_ERROR;
}

static cgrad_error gelu_forward_dispatch(const struct tensor *const x, struct tensor *const out)
{
    return gelu_forward_scalar(x, out);
}

static cgrad_error gelu_forward_scalar(const struct tensor *const x, struct tensor *const out)
{
    switch (x->dtype)
    {
    case DTYPE_FLOAT64:
        return gelu_forward_scalar_f64(x, out);

    case DTYPE_FLOAT32:
        return gelu_forward_scalar_f32(x, out);

    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error gelu_forward_scalar_f64(const struct tensor *const x, struct tensor *const out)
{
    const double *x_data = (const double *)x->data;
    double *out_data = (double *)out->data;

    for (size_t i = 0; i < out->data_size; i++)
    {
        const double x_value = x_data[i];
        out_data[i] = 0.5 * x_value * (1.0 + erf(x_value * GELU_INV_SQRT_2));
    }

    return NO_ERROR;
}

static cgrad_error gelu_forward_scalar_f32(const struct tensor *const x, struct tensor *const out)
{
    const float *x_data = (const float *)x->data;
    float *out_data = (float *)out->data;

    const float inv_sqrt_2 = (float)GELU_INV_SQRT_2;

    for (size_t i = 0; i < out->data_size; i++)
    {
        const float x_value = x_data[i];
        out_data[i] = 0.5f * x_value * (1.0f + erff(x_value * inv_sqrt_2));
    }

    return NO_ERROR;
}