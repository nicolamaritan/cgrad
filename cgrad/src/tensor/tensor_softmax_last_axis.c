#include "cgrad/tensor/tensor_softmax_last_axis.h"
#include "cgrad/tensor/tensor_copy.h"
#include "cgrad/tensor/tensor_helpers.h"
#include "cgrad/tensor/tensor_equality.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include <math.h>

typedef enum tensor_softmax_last_axis_operand
{
    TENSOR,
} tensor_softmax_last_axis_operand;

static inline cgrad_error tensor_softmax_last_axis_update_graph(struct tensor *const t, struct tensor **const out, struct cgrad_env *const env);

static inline cgrad_error tensor_softmax_last_axis_dispatch(const struct tensor *const t, struct tensor *const out);

static cgrad_error tensor_softmax_last_axis_f64(const struct tensor *const t, struct tensor *const out);

static cgrad_error tensor_softmax_last_axis_f32(const struct tensor *const t, struct tensor *const out);

static cgrad_error tensor_softmax_last_axis_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

static cgrad_error tensor_softmax_last_axis_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

static cgrad_error tensor_softmax_last_axis_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

// Helpers
static inline size_t get_samples_number(const struct tensor *const t);

cgrad_error tensor_softmax_last_axis(struct tensor *const t, struct tensor **const out, const bool track_grad, struct cgrad_env *const env)
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

    cgrad_error err = tensor_softmax_last_axis_dispatch(t, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor_softmax_last_axis_update_graph(t, out, env);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_softmax_last_axis_update_graph(struct tensor *const t, struct tensor **const out, struct cgrad_env *const env)
{
    return add_computational_graph_link(t, TENSOR, *out, &tensor_softmax_last_axis_backpropagate, env);
}

static inline cgrad_error tensor_softmax_last_axis_dispatch(const struct tensor *const t, struct tensor *const out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor_softmax_last_axis_f64(t, out);
    case DTYPE_FLOAT32:
        return tensor_softmax_last_axis_f32(t, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_softmax_last_axis_f64(const struct tensor *const t, struct tensor *const out)
{
    double *restrict out_data = (double *)out->data;
    double *restrict t_data = (double *)t->data;

    size_t num_samples = get_samples_number(t);
    size_t sample_size = t->shape[t->shape_size - 1];

    for (size_t i = 0; i < num_samples; i++)
    {
        double max = -INFINITY;
        double denom = 0.0;
        for (size_t j = 0; j < sample_size; j++)
        {
            double sample = t_data[i * sample_size + j];
            if (sample > max)
            {
                denom *= exp(max - sample);
                max = sample;
            }
            denom += exp(sample - max);
        }
        for (size_t j = 0; j < sample_size; j++)
        {
            out_data[i * sample_size + j] = exp(t_data[i * sample_size + j] - max) / denom;
        }
    }

    return NO_ERROR;
}

static cgrad_error tensor_softmax_last_axis_f32(const struct tensor *const t, struct tensor *const out)
{
    float *restrict out_data = (float *)out->data;
    float *restrict t_data = (float *)t->data;

    size_t num_samples = get_samples_number(t);
    size_t sample_size = t->shape[t->shape_size - 1];

    for (size_t i = 0; i < num_samples; i++)
    {
        float max = -INFINITY;
        float denom = 0.0f;
        for (size_t j = 0; j < sample_size; j++)
        {
            float sample = t_data[i * sample_size + j];
            if (sample > max)
            {
                denom *= expf(max - sample);
                max = sample;
            }
            denom += expf(sample - max);
        }
        for (size_t j = 0; j < sample_size; j++)
        {
            out_data[i * sample_size + j] = expf(t_data[i * sample_size + j] - max) / denom;
        }
    }

    return NO_ERROR;
}

static inline size_t get_samples_number(const struct tensor *const t)
{
    if (t->shape_size == 0)
    {
        return 0; 
    }

    size_t num_samples = 1;
    for (size_t i = 0; i < t->shape_size - 1; i++)
    {
        num_samples *= t->shape[i];
    }
    return num_samples;
}

static cgrad_error tensor_softmax_last_axis_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor_softmax_last_axis_backpropagate_f64(ctx, grad_wrt_out, grad_wrt_operand);
    case DTYPE_FLOAT32:
        return tensor_softmax_last_axis_backpropagate_f32(ctx, grad_wrt_out, grad_wrt_operand);
    default:
        return AUTOGRAD_BACKPROPAGATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_softmax_last_axis_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *t = ctx->operands[TENSOR];
    double *restrict t_data = (double *)t->data;
    double *restrict grad_out_data = (double *)grad_wrt_out->data;
    double *restrict grad_in_data = (double *)grad_wrt_operand->data;

    size_t num_samples = get_samples_number(t);
    size_t sample_size = t->shape[t->shape_size - 1];

    for (size_t i = 0; i < num_samples; i++)
    {
        double max = -INFINITY;
        double denom = 0.0;
        for (size_t j = 0; j < sample_size; j++)
        {
            double sample = t_data[i * sample_size + j];
            if (sample > max)
            {
                denom *= exp(max - sample);
                max = sample;
            }
            denom += exp(sample - max);
        }

        double dot = 0.0;
        for (size_t j = 0; j < sample_size; j++)
        {
            double y_j = exp(t_data[i * sample_size + j] - max) / denom;
            dot += y_j * grad_out_data[i * sample_size + j];
        }

        for (size_t j = 0; j < sample_size; j++)
        {
            double y_j = exp(t_data[i * sample_size + j] - max) / denom;
            grad_in_data[i * sample_size + j] = y_j * (grad_out_data[i * sample_size + j] - dot);
        }
    }

    return NO_ERROR;
}

static cgrad_error tensor_softmax_last_axis_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *t = ctx->operands[TENSOR];
    float *restrict t_data = (float *)t->data;
    float *restrict grad_out_data = (float *)grad_wrt_out->data;
    float *restrict grad_in_data = (float *)grad_wrt_operand->data;

    size_t num_samples = get_samples_number(t);
    size_t sample_size = t->shape[t->shape_size - 1];

    for (size_t i = 0; i < num_samples; i++)
    {
        float max = -INFINITY;
        float denom = 0.0f;
        for (size_t j = 0; j < sample_size; j++)
        {
            float sample = t_data[i * sample_size + j];
            if (sample > max)
            {
                denom *= expf(max - sample);
                max = sample;
            }
            denom += expf(sample - max);
        }

        float dot = 0.0f;
        for (size_t j = 0; j < sample_size; j++)
        {
            float y_j = expf(t_data[i * sample_size + j] - max) / denom;
            dot += y_j * grad_out_data[i * sample_size + j];
        }

        for (size_t j = 0; j < sample_size; j++)
        {
            float y_j = expf(t_data[i * sample_size + j] - max) / denom;
            grad_in_data[i * sample_size + j] = y_j * (grad_out_data[i * sample_size + j] - dot);
        }
    }

    return NO_ERROR;
}