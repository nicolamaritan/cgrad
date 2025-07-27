#include "layers/relu.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include "utils/simd_support.h"
#include <stdlib.h>
#include <stdio.h>

#if SIMD_AVX_LEVEL > SIMD_AVX_LEVEL_0
#include <immintrin.h>
#endif

typedef enum relu_layer_operand
{
    RELU_ONLY_OPERAND,
} relu_layer_operand;

static void relu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void relu_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void relu_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error relu_forward_dispatch(const struct tensor *const x, struct tensor *const out);
#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
static cgrad_error relu_forward_dispatch_avx_256(const struct tensor *const x, struct tensor *const out);
static void relu_forward_unchecked_avx_256_f64(const struct tensor *const x, struct tensor *const out);
static void relu_forward_unchecked_avx_256_f32(const struct tensor *const x, struct tensor *const out);
#else
static cgrad_error relu_forward_dispatch_scalar(const struct tensor *const x, struct tensor *const out);
static void relu_forward_unchecked_scalar_f64(const struct tensor *const x, struct tensor *const out);
static void relu_forward_unchecked_scalar_f32(const struct tensor *const x, struct tensor *const out);
#endif

    cgrad_error relu_forward_graph(struct tensor *const x, struct tensor *const out, struct autograd_allocators *ag_allocators)
{
    cgrad_error error = relu_forward(x, out);
    if (error != NO_ERROR)
    {
        return error;
    }

    error = add_computational_graph_link(x, RELU_ONLY_OPERAND, out, &relu_backpropagate, ag_allocators);
    return error;
}

cgrad_error relu_forward(const struct tensor *const x, struct tensor *const out)
{
    if (!x || !out)
    {
        return TENSOR_NULL;
    }
    if (!x->data || !out->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (!tensor_same_shape(x, out))
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    return relu_forward_dispatch(x, out);
}

static void relu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{

    /*
        Gradient computation of dz/dX.
        dz/dX is the Hadamard Product of grad_wrt_out = dz/drelu(X) and drelu(X)/dX,
        since element (i, j) of relu(X) depends only on element (i, j) of X.
    */

    switch (grad_wrt_operand->cgrad_dtype)
    {
    case DTYPE_FLOAT64:
        relu_backpropagate_f64(ctx, grad_wrt_out, grad_wrt_operand);
        break;
    case DTYPE_FLOAT32:
        relu_backpropagate_f32(ctx, grad_wrt_out, grad_wrt_operand);
        break;
    default:
        break;
    }
}

static void relu_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *const x = ctx->operands[RELU_ONLY_OPERAND];

    // Avoid multiple indirections for performance
    double *x_data = (double *)x->data;
    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;
    double *grad_wrt_out_data = (double *)grad_wrt_out->data;
    size_t grad_wrt_operand_data_size = grad_wrt_operand->data_size;

    for (size_t i = 0; i < grad_wrt_operand_data_size; i++)
    {
        // Element wise product
        grad_wrt_operand_data[i] = (x_data[i] > 0 ? 1 : 0) * grad_wrt_out_data[i];
    }
}

static void relu_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *const x = ctx->operands[RELU_ONLY_OPERAND];

    // Avoid multiple indirections for performance
    float *x_data = (float *)x->data;
    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    float *grad_wrt_out_data = (float *)grad_wrt_out->data;
    size_t grad_wrt_operand_data_size = grad_wrt_operand->data_size;

    for (size_t i = 0; i < grad_wrt_operand_data_size; i++)
    {
        // Element wise product
        grad_wrt_operand_data[i] = (x_data[i] > 0 ? 1 : 0) * grad_wrt_out_data[i];
    }
}

static cgrad_error relu_forward_dispatch(const struct tensor *const x, struct tensor *const out)
{
#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
    return relu_forward_dispatch_avx_256(x, out);
#else
    return relu_forward_dispatch_scalar(x, out);
#endif
}

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
static cgrad_error relu_forward_dispatch_avx_256(const struct tensor *const x, struct tensor *const out)
{
    switch (x->cgrad_dtype)
    {
    case DTYPE_FLOAT64:
        relu_forward_unchecked_avx_256_f64(x, out);
        break;
    case DTYPE_FLOAT32:
        relu_forward_unchecked_avx_256_f32(x, out);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
        break;
    }

    return NO_ERROR;
}

static void relu_forward_unchecked_avx_256_f64(const struct tensor *const x, struct tensor *const out)
{
    const size_t PARALLELIZED_ITEMS = sizeof(__m256d) / sizeof(double);

    double zeros[PARALLELIZED_ITEMS];
    memset(zeros, 0, sizeof(zeros));
    __m256d zeros_vals = _mm256_load_pd(zeros);

    double *x_data = (double *)x->data;
    double *out_data = (double *)out->data;

    size_t i = 0;
    for (; i + PARALLELIZED_ITEMS - 1 < x->data_size; i += PARALLELIZED_ITEMS)
    {
        __m256d x_vals = _mm256_load_pd(&x_data[i]);
        __m256d relu_vals = _mm256_max_pd(zeros_vals, x_vals);
        _mm256_store_pd(&out_data[i], relu_vals);
    }

    // Handle remaining items
    for (; i < x->data_size; i++)
    {
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }
}

static void relu_forward_unchecked_avx_256_f32(const struct tensor *const x, struct tensor *const out)
{
    const size_t PARALLELIZED_ITEMS = sizeof(__m256) / sizeof(float);

    float zeros[PARALLELIZED_ITEMS];
    memset(zeros, 0, sizeof(zeros));
    __m256 zeros_vals = _mm256_load_ps(zeros);

    float *x_data = (float *)x->data;
    float *out_data = (float *)out->data;

    size_t i = 0;
    for (; i + PARALLELIZED_ITEMS - 1 < x->data_size; i += PARALLELIZED_ITEMS)
    {
        __m256 x_vals = _mm256_load_ps(&x_data[i]);
        __m256 relu_vals = _mm256_max_ps(zeros_vals, x_vals);
        _mm256_storeu_ps(&out_data[i], relu_vals);
    }

    // Handle remaining items
    for (; i < x->data_size; i++)
    {
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }
}
#else
static cgrad_error relu_forward_dispatch_scalar(const struct tensor *const x, struct tensor *const out)
{
    switch (x->cgrad_dtype)
    {
    case DTYPE_FLOAT64:
        relu_forward_unchecked_scalar_f64(x, out);
        break;
    case DTYPE_FLOAT32:
        relu_forward_unchecked_scalar_f32(x, out);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void relu_forward_unchecked_scalar_f64(const struct tensor *const x, struct tensor *const out)
{
    double *x_data = (double *)x->data;
    double *out_data = (double *)out->data;
    for (size_t i = 0; i < out->data_size; i++)
    {
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }
}

static void relu_forward_unchecked_scalar_f32(const struct tensor *const x, struct tensor *const out)
{
    float *x_data = (float *)x->data;
    float *out_data = (float *)out->data;
    for (size_t i = 0; i < out->data_size; i++)
    {
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }
}
#endif