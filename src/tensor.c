#include "tensor.h"
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

tensor *tensor_alloc(size_t *shape, size_t shape_size)
{
    tensor *t = tensor_no_grad_alloc(shape, shape_size);
    t->grad = tensor_no_grad_zero_alloc(shape, shape_size);
    return t;
}

tensor* tensor_no_grad_alloc(size_t *shape, size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
       data_size *= shape[i];
    }

    tensor *t = (tensor *)malloc(sizeof(tensor));
    double* data = (double *)malloc(data_size * sizeof(double));
    size_t *_shape = (size_t *)malloc((shape_size + 1)* sizeof(size_t));

    // Init _shape
    memcpy(_shape, shape, sizeof(size_t) * shape_size);
    _shape[shape_size] = 0;

    t->data = data;
    t->shape = _shape;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;

    return t;
}

tensor* tensor_no_grad_zero_alloc(size_t *shape, size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
        data_size *= shape[i];

    tensor *t = (tensor *)malloc(sizeof(tensor));
    double* data = (double *)calloc(data_size, sizeof(double));
    size_t *_shape = (size_t *)malloc((shape_size + 1)* sizeof(size_t));

    // Init _shape
    memcpy(_shape, shape, shape_size);
    _shape[shape_size] = 0;


    t->data = data;
    t->shape = _shape;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;

    return t;
}

tensor *tensor2d_alloc(size_t rows, size_t cols)
{
    tensor *t = tensor2d_no_grad_alloc(rows, cols);
    t->grad = tensor2d_no_grad_zero_alloc(rows, cols);
    return t;
}

tensor *tensor2d_no_grad_alloc(size_t rows, size_t cols)
{
    tensor *t = (tensor *)malloc(sizeof(tensor));
    double *data = (double *)malloc(rows * cols * sizeof(double));
    size_t *shape = (size_t *)malloc(3 * sizeof(size_t));
    shape[2] = 0;

    if (!shape || !data || !t)
    {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return NULL;
    }

    // Set the 2 dimensions
    shape[0] = rows;
    shape[1] = cols;

    t->data = data;
    t->shape = shape;
    t->node = NULL;
    t->data_size = rows * cols;
    t->shape_size = 2;
    t->grad = NULL;

    return t;
}

tensor *tensor2d_no_grad_zero_alloc(size_t rows, size_t cols)
{
    tensor *t = (tensor *)malloc(sizeof(tensor));
    double *data = (double *)calloc(rows * cols, sizeof(double)); // Ensure 0 for all cells
    size_t *shape = (size_t *)malloc(3 * sizeof(size_t));
    shape[2] = 0;

    if (!shape || !data || !t)
    {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return NULL;
    }

    // Set the 2 dimensions
    shape[0] = rows;
    shape[1] = cols;

    t->data = data;
    t->shape = shape;
    t->node = NULL;
    t->data_size = rows * cols;
    t->shape_size = 2;
    t->grad = NULL;

    return t;
}

tensor *tensor2d_alloc_like(tensor *t)
{
    if (!t->shape)
        return NULL;
    if (!t->shape[0] || !t->shape[1])
        return NULL;
    return tensor2d_alloc(t->shape[0], t->shape[1]);
}

void tensor_free(tensor *t)
{
    free(t->data);
    free(t->shape);

    if (t->grad)
    {
        tensor_no_grad_free(t->grad);
    }

    free(t);
}

void tensor_no_grad_free(tensor *t)
{
    free(t->data);
    free(t->shape);
    free(t);
}

tensor_error tensor2d_mult(const tensor *const A, const tensor *const B, tensor *const out)
{
    if (!A || !B || !out)
        return TENSOR_NULL;
    if (!A->shape || !B->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (A->shape[1] != B->shape[0])
        return TENSOR_SHAPE_MISMATCH; // Columns of A != rows of B
    if (out->shape[0] != A->shape[0] || out->shape[1] != B->shape[1])
        return TENSOR_SHAPE_MISMATCH; // Output shape mismatch

    tensor2d_mult_unchecked(A, B, out);
    return TENSOR_OK;
}

void tensor2d_mult_unchecked(const tensor *const A, const tensor *const B, tensor *const out)
{
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A->shape[0], // M
        B->shape[1], // N
        A->shape[1], // K (must match B->shape[0])
        1.0,
        A->data,
        A->shape[1], // lda
        B->data,
        B->shape[1], // ldb
        0.0,
        out->data,
        out->shape[1] // ldc
    );
}

tensor_error tensor2d_trans(const tensor *const t, tensor *const out)
{
    if (!t || !out)
        return TENSOR_NULL;
    if (!t->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (!t->data || !out->data)
        return TENSOR_DATA_NULL;
    if (t->shape[0] != out->shape[1] || t->shape[1] != out->shape[0])
        return TENSOR_SHAPE_MISMATCH;

    tensor2d_trans_unchecked(t, out);
    return TENSOR_OK;
}

void tensor2d_trans_unchecked(const tensor *const t, tensor *const out)
{
    // Extract the shape of the input tensor (t)
    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    out->shape[0] = cols;
    out->shape[1] = rows;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            out->data[j * rows + i] = t->data[i * cols + j];
        }
    }
}

void tensor_copy(const tensor *const src, tensor *dest)
{
    if (!src || !dest)
        return;

    // Ensure shapes are identical before copying
    if (src->shape[0] == dest->shape[0] && src->shape[1] == dest->shape[1])
    {
        memcpy(dest->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    }
    else
    {
        fprintf(stderr, "Error: Tensor shapes do not match for copy operation.\n");
    }
}

// Function to create a copy of a tensor and return a new instance
tensor *tensor_clone(const tensor *const src)
{
    if (!src)
        return NULL;

    tensor *new_tensor = tensor2d_alloc(src->shape[0], src->shape[1]);
    if (!new_tensor)
        return NULL;

    memcpy(new_tensor->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    return new_tensor;
}

tensor_error tensor2d_add_row_vector(tensor *const A, const tensor *const v)
{
    if (!A || !v)
        return TENSOR_NULL;
    if (!A->data || !v->data)
        return TENSOR_DATA_NULL;
    if (!A->shape || !v->shape)
        return TENSOR_SHAPE_NULL;
    if (A->shape_size != 2 || v->shape_size != 2)
        return TENSOR_WRONG_SHAPE;
    if (v->shape[1] != 1)
        return TENSOR_WRONG_SHAPE;
    if (A->shape[1] != v->shape[0])
        return TENSOR_SHAPE_MISMATCH;

    tensor2d_add_row_vector_unchecked(A, v);
    return TENSOR_OK;
}

void tensor2d_add_row_vector_unchecked(tensor *const A, const tensor *const v)
{
    size_t rows = A->shape[0]; // Number of rows
    size_t cols = A->shape[1]; // Number of columns

    if (v->shape[0] != cols || v->shape[1] != 1)
    {
        // Handle dimension mismatch (bias should have shape [out_dim, 1])
        return;
    }

    double *out_data = A->data;
    double *bias_data = v->data;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            out_data[i * cols + j] += bias_data[j];
        }
    }
}

void tensor_add_unchecked(const tensor *const A, const tensor *const B, tensor *const out)
{
    for (size_t i = 0; i < A->data_size; i++)
    {
        out->data[i] = A->data[i] + B->data[i];
    }
}

tensor_error tensor_add(const tensor *const A, const tensor *const B, tensor *const out)
{
    if (!A || !B || !out)
        return TENSOR_NULL;
    if (!A->data || !B->data || !out->data)
        return TENSOR_DATA_NULL;
    if (!A->shape || !B->shape || !out->shape)
        return TENSOR_SHAPE_NULL;
    if (A->data_size != B->data_size || B->data_size != out->data_size)
        return TENSOR_DATA_SIZE_MISMATCH;
    if (tensor_same_shape(A, B))
        return TENSOR_SHAPE_MISMATCH;

    tensor_add_unchecked(A, B, out);
    return TENSOR_OK;
}

tensor_error tensor_add_inplace(tensor *A, const tensor *const B)
{
    if (!A || !B)
        return TENSOR_NULL;
    if (!A->data || !B->data)
        return TENSOR_DATA_NULL;
    if (!A->shape || !B->shape)
        return TENSOR_SHAPE_NULL;
    if (A->data_size != B->data_size)
        return TENSOR_DATA_SIZE_MISMATCH;
    if (!tensor_same_shape(A, B))
        return false;

    tensor_add_inplace_unchecked(A, B);
    return TENSOR_OK;
}

void tensor_add_inplace_unchecked(tensor *A, const tensor *const B)
{
    for (size_t i = 0; i < A->data_size; i++)
    {
        A->data[i] += B->data[i];
    }
}

bool tensor_same_shape(const tensor *const A, const tensor *const B)
{
    if (A->shape_size != B->shape_size)
    {
        return false;
    }

    size_t shape_size = A->shape_size;
    for (size_t i = 0; i < shape_size; i++)
    {
        if (A->shape[i] != B->shape[i])
        {
            return false;
        }
    }
    return true;
}

void print_tensor_recursive(const double *data, const size_t *shape, size_t dimensions, size_t offset)
{
    if (dimensions == 1)
    {
        printf("[");
        for (size_t i = 0; i < shape[0]; i++)
        {
            printf("%lf", data[offset + i]);
            if (i < shape[0] - 1)
            {
                printf(", ");
            }
        }
        printf("]");
        return;
    }

    printf("[");
    for (size_t i = 0; i < shape[0]; i++)
    {
        print_tensor_recursive(data, shape + 1, dimensions - 1, offset + i * shape[1]);
        if (i < shape[0] - 1)
        {
            printf(", ");
        }
    }
    printf("]");
}

void print_tensor(const tensor *const t)
{
    if (t == NULL || t->data == NULL || t->shape == NULL)
    {
        printf("Invalid tensor\n");
        return;
    }
    size_t dimensions = 0;
    while (t->shape[dimensions] != 0)
    {
        dimensions++;
    }
    print_tensor_recursive(t->data, t->shape, dimensions, 0);
    printf("\n");
}
