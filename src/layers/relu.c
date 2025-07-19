#include "layers/relu.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/computational_graph/computational_graph_link.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef USE_AVX
    #include <immintrin.h>
    #define HAS_AVX_SUPPORT 1
#else
    #define HAS_AVX_SUPPORT 0
#endif

typedef enum relu_layer_operand
{
    RELU_ONLY_OPERAND,
} relu_layer_operand;

static void relu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
void relu_forward_unchecked(const struct tensor *const x, struct tensor *const out);
#if HAS_AVX_SUPPORT
    void relu_forward_unchecked_avx(const struct tensor* const x, struct tensor* const out);
#else
    void relu_forward_unchecked_sequential(const struct tensor* const x, struct tensor* const out);
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

cgrad_error relu_forward(const struct tensor* const x, struct tensor* const out)
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

    relu_forward_unchecked(x, out);
    return NO_ERROR;

    // Avoid multiple indirections for performance
    // double* x_data = x->data;
    // double* out_data = out->data;
    // size_t out_data_size = out->data_size;

    // for (size_t i = 0; i < out_data_size; i++)
    // {
    //     out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    // }

    // return NO_ERROR;
}

static void relu_backpropagate(const struct backpropagation_context *const ctx, const struct tensor* const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    const struct tensor *const x = ctx->operands[RELU_ONLY_OPERAND];
    
    // Avoid multiple indirections for performance
    double* x_data = x->data;
    double* grad_wrt_operand_data = grad_wrt_operand->data;
    size_t grad_wrt_operand_data_size = grad_wrt_operand->data_size;
    
    /*
        Gradient computation of dz/dX.
        dz/dX is the Hadamard Product of grad_wrt_out = dz/drelu(X) and drelu(X)/dX,
        since element (i, j) of relu(X) depends only on element (i, j) of X.
    */
    
    for (size_t i = 0; i < grad_wrt_operand_data_size; i++)
    {
        // Element wise product
        grad_wrt_operand_data[i] = (x_data[i] > 0 ? 1 : 0) * grad_wrt_out->data[i];
    }
}

void relu_forward_unchecked(const struct tensor *const x, struct tensor *const out)
{
    #if HAS_AVX_SUPPORT
        relu_forward_unchecked_avx(x, out);
    #else
        relu_forward_unchecked_sequential(x, out);
    #endif
}

#if HAS_AVX_SUPPORT
    #define AVX_DOUBLE_NUMBER 4
    void relu_forward_unchecked_avx(const struct tensor* const x, struct tensor* const out)
    {
        double zeros[AVX_DOUBLE_NUMBER];
        memset(zeros, 0, HAS_AVX_SUPPORT);
        __m256d zeros_vals = _mm256_loadu_pd(zeros);

        size_t i = 0;
        for (; i + AVX_DOUBLE_NUMBER - 1 < x->data_size; i += AVX_DOUBLE_NUMBER)
        {
            __m256d x_vals = _mm256_loadu_pd(&x->data[i]);
            __m256d relu_vals = _mm256_max_pd(zeros_vals, x_vals);
            _mm256_storeu_pd(&out->data[i], relu_vals);
        }

        for (; i < x->data_size; i++)
        {
            out->data[i] = x->data[i] > 0 ? x->data[i] : 0;
        }
    }
#else
    void relu_forward_unchecked_sequential(const struct tensor* const x, struct tensor* const out)
    {
        for (size_t i = 0; i < out->data_size; i++)
        {
            out->data[i] = x->data[i] > 0 ? x->data[i] : 0;
        }
    }
#endif