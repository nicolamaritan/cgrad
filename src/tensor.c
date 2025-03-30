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

void tensor2d_copy(const tensor *const src, tensor *const dest)
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

void tensor_copy(const tensor *const src, tensor *const dest)
{
    if (!src || !dest)
        return;
    
    if (src->shape_size != dest->shape_size)
    {
        fprintf(stderr, "Error: Tensor shapes do not match for copy operation.\n");
        return;
    }

    for (size_t i = 0; i < src->shape_size; i++)
    {
        if (src->shape[i] != dest->shape[i])
        {
            fprintf(stderr, "Error: Tensor shapes do not match for copy operation.\n");
            return;
        }
    }

    memcpy(dest->data, src->data, sizeof(double) * src->data_size);
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

void tensor_fill(tensor *const t, double value)
{
    if (!t || !t->data)
        return;

    size_t data_size = t->data_size;
    for (size_t i = 0; i < data_size; i++)
    {
        t->data[i] = value;
    }
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
