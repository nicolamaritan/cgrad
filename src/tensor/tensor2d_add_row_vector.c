#include "tensor/tensor2d_add_row_vector.h"
#include "tensor/tensor_sum.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include "utils/simd_support.h"

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_0
#include <immintrin.h>
#endif

typedef enum tensor2d_add_row_vector_operand
{
    TENSOR2D,
    ROW_VECTOR,
} tensor2d_add_row_vector_operand;

static inline cgrad_error tensor2d_add_row_vector_dispatch(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static cgrad_error tensor2d_add_row_vector_backpropagate_tensor2d(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error tensor2d_add_row_vector_backpropagate_row_vector(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
static cgrad_error tensor2d_add_row_vector_dispatch_avx_256(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static cgrad_error tensor2d_add_row_vector_avx_256_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static cgrad_error tensor2d_add_row_vector_avx_256_f32(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
#else
static cgrad_error tensor2d_add_row_vector_dispatch_scalar(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static cgrad_error tensor2d_add_row_vector_scalar_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
static cgrad_error tensor2d_add_row_vector_scalar_f32(const struct tensor *const t, const struct tensor *const v, struct tensor *out);
#endif

cgrad_error tensor2d_add_row_vector(const struct tensor *const t, const struct tensor *const v, struct tensor **const out, struct tensor_allocator *const tensor_alloc)
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

    (*out) = tensor_allocator_alloc(tensor_alloc, t->shape, t->shape_size, t->dtype);

    if (!(*out))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    return tensor2d_add_row_vector_dispatch(t, v, *out);
}

cgrad_error tensor2d_add_row_vector_graph(struct tensor *const t, struct tensor *const v, struct tensor **const out, struct allocators *const allocs)
{
    cgrad_error err = tensor2d_add_row_vector(t, v, out, allocs->tensor_alloc);
    if (err != NO_ERROR)
    {
        return err;
    }

    // Update computational graph
    err = add_computational_graph_link(t, TENSOR2D, *out, &tensor2d_add_row_vector_backpropagate_tensor2d, allocs);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(v, ROW_VECTOR, *out, &tensor2d_add_row_vector_backpropagate_row_vector, allocs);

    return err;
}

cgrad_error tensor2d_add_row_vector_into(const struct tensor *const t, const struct tensor *const v, struct tensor *const out)
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

static inline cgrad_error tensor2d_add_row_vector_dispatch(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
    return tensor2d_add_row_vector_dispatch_avx_256(t, v, out);
#else
    return tensor2d_add_row_vector_dispatch_scalar(t, v, out);
#endif
}

static cgrad_error tensor2d_add_row_vector_backpropagate_tensor2d(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    cgrad_error err = tensor2d_copy(grad_wrt_out, grad_wrt_operand);
    if (err != NO_ERROR)
    {
        return err;
    }

    return NO_ERROR;
}

static cgrad_error tensor2d_add_row_vector_backpropagate_row_vector(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    cgrad_error err = tensor_sum(grad_wrt_out, 0, grad_wrt_operand);
    if (err != NO_ERROR)
    {
        return err;
    }

    return NO_ERROR;
}

#if SIMD_AVX_LEVEL >= SIMD_AVX_LEVEL_256
static cgrad_error tensor2d_add_row_vector_dispatch_avx_256(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor2d_add_row_vector_avx_256_f64(t, v, out);
    case DTYPE_FLOAT32:
        return tensor2d_add_row_vector_avx_256_f32(t, v, out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor2d_add_row_vector_avx_256_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
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

        /**
         * Perform SIMD loop only if memory is 32-byte aligned. Even if the first row of the tensor
         * is aligned to 32-byte, if the number of columns is not a multiple of 32 bytes, other rows
         * may not be properly aligned.
         */

        if (row_offset % sizeof(__m256d) == 0)
        {
            for (; j + PARALLELIZED_ITEMS - 1 < cols; j += PARALLELIZED_ITEMS)
            {
                __m256d a_vals = _mm256_load_pd(&t_data[row_offset + j]);
                __m256d v_vals = _mm256_load_pd(&v_data[j]);
                __m256d sum = _mm256_add_pd(a_vals, v_vals);
                _mm256_store_pd(&out_data[row_offset + j], sum);
            }
        }

        for (; j < cols; j++)
        {
            out_data[row_offset + j] = t_data[row_offset + j] + v_data[j];
        }
    }

    return NO_ERROR;
}

static cgrad_error tensor2d_add_row_vector_avx_256_f32(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    float *t_data = (float *)t->data;
    float *v_data = (float *)v->data;
    float *out_data = (float *)out->data;

    const size_t PARALLELIZED_ITEMS = sizeof(__m256) / sizeof(float);

    for (size_t i = 0; i < rows; i++)
    {
        size_t row_offset = i * cols;
        size_t j = 0;

        // Same motivation for f64 version
        if (row_offset % sizeof(__m256) == 0)
        {
            for (; j + PARALLELIZED_ITEMS - 1 < cols; j += PARALLELIZED_ITEMS)
            {
                __m256 a_vals = _mm256_load_ps(&t_data[row_offset + j]);
                __m256 v_vals = _mm256_load_ps(&v_data[j]);
                __m256 sum = _mm256_add_ps(a_vals, v_vals);
                _mm256_store_ps(&out_data[row_offset + j], sum);
            }
        }

        for (; j < cols; j++)
        {
            out_data[row_offset + j] = t_data[row_offset + j] + v_data[j];
        }
    }

    return NO_ERROR;
}
#else
static cgrad_error tensor2d_add_row_vector_dispatch_scalar(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    switch (t->cgrad_dtype)
    {
    case DTYPE_FLOAT64:
        return tensor2d_add_row_vector_scalar_f64(t, v, out);
    case DTYPE_FLOAT32:
        return tensor2d_add_row_vector_scalar_f32(t, v, out);
    default:
        return AUTOGRAD_BACKPROPAGATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor2d_add_row_vector_scalar_f64(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
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

    return NO_ERROR;
}

static cgrad_error tensor2d_add_row_vector_scalar_f32(const struct tensor *const t, const struct tensor *const v, struct tensor *out)
{
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    float *t_data = (float *)t->data;
    float *v_data = (float *)v->data;
    float *out_data = (float *)out->data;

    for (size_t i = 0; i < rows; i++)
    {
        size_t row_offset = i * cols;

        for (size_t j = 0; j < cols; j++)
        {
            out_data[row_offset + j] = t_data[row_offset + j] + v_data[j];
        }
    }

    return NO_ERROR;
}
#endif