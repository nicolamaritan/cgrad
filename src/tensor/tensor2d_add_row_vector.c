#include "tensor/tensor2d_add_row_vector.h"
#include "tensor/tensor_sum.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include "utils/simd_support.h"
#include <stdlib.h>

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_0
#include <immintrin.h>
#endif

typedef enum tensor2d_add_row_vector_operand
{
    TENSOR2D,
    ROW_VECTOR,
} tensor2d_add_row_vector_operand;

static cgrad_error tensor2d_add_row_vector_dispatch(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static void tensor2d_add_row_vector_backpropagate_tensor2d(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static void tensor2d_add_row_vector_backpropagate_row_vector(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
static cgrad_error tensor2d_add_row_vector_dispatch_avx_256(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static void tensor2d_add_row_vector_unchecked_avx_256_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
#else
static cgrad_error tensor2d_add_row_vector_dispatch_scalar(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static void tensor2d_add_row_vector_unchecked_scalar_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
#endif

cgrad_error tensor2d_add_row_vector(const struct tensor *const t, const struct tensor *const v, struct tensor *const out)
{
    if (!t || !v)
    {
        return TENSOR_NULL;
    }
    if (!t->data || !v->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (t->shape_size != 2 || v->shape_size != 2)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (v->shape[1] != 1)
    {
        return TENSOR_WRONG_SHAPE;
    }
    if (t->shape[1] != v->shape[0])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (t->dtype != v->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return tensor2d_add_row_vector_dispatch(t, v, out);
}

cgrad_error tensor2d_add_row_vector_graph(struct tensor *const t, struct tensor *const v, struct tensor *const out, struct autograd_allocators *allocators)
{
    cgrad_error err = tensor2d_add_row_vector(t, v, out);
    if (err != NO_ERROR)
    {
        return err;
    }

    // Update computational graph
    err = add_computational_graph_link(t, TENSOR2D, out, &tensor2d_add_row_vector_backpropagate_tensor2d, allocators);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(v, ROW_VECTOR, out, &tensor2d_add_row_vector_backpropagate_row_vector, allocators);

    return err;
}

static cgrad_error tensor2d_add_row_vector_dispatch(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
    return tensor2d_add_row_vector_dispatch_avx_256(t, v, out);
#else
    return tensor2d_add_row_vector_dispatch_scalar(t, v, out);
#endif
}

static void tensor2d_add_row_vector_backpropagate_tensor2d(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    tensor2d_copy(grad_wrt_out, grad_wrt_operand);
}

static void tensor2d_add_row_vector_backpropagate_row_vector(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    tensor_sum(grad_wrt_out, 0, grad_wrt_operand);
}

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
static cgrad_error tensor2d_add_row_vector_dispatch_avx_256(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        tensor2d_add_row_vector_unchecked_avx_256_f64(t, v, out);
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
        break;
    }

    return NO_ERROR;
}

static void tensor2d_add_row_vector_unchecked_avx_256_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    double *t_data = (double *)t->data;
    double *v_data = (double *)v->data;
    double *out_data = (double *)out->data;

    const size_t PARALLELIZED_ITEMS = sizeof(__m256d) / sizeof(double);

    for (size_t i = 0; i < rows; i++)
    {
        size_t row_offset = i * cols;
        size_t j = 0;

        for (; j + PARALLELIZED_ITEMS - 1 < cols; j += PARALLELIZED_ITEMS)
        {
            __m256d a_vals = _mm256_loadu_pd(&t_data[row_offset + j]);
            __m256d v_vals = _mm256_loadu_pd(&v_data[j]);
            __m256d sum = _mm256_add_pd(a_vals, v_vals);
            _mm256_storeu_pd(&out_data[row_offset + j], sum);
        }

        for (; j < cols; j++)
        {
            out_data[row_offset + j] = t_data[row_offset + j] + v_data[j];
        }
    }
}
#else
static cgrad_error tensor2d_add_row_vector_dispatch_scalar(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        tensor2d_add_row_vector_unchecked_scalar_f64(t, v, out);
        break;
    default:
        return TENSOR_OPERATION_DTYPE_NOT_SUPPORTED;
    }

    return NO_ERROR;
}

static void tensor2d_add_row_vector_unchecked_scalar_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    double *t_data = (double *)t->data;
    double *v_data = (double *)v->data;
    double *out_data = (double *)out->data;

    for (size_t i = 0; i < rows; i++)
    {
        size_t row_offset = i * cols;

        for (size_t j = 0; j < cols; j++)
        {
            out_data[row_offset + j] = t_data[row_offset + j] + v_data[j];
        }
    }
}
#endif