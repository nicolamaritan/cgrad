#include "tensor/tensor.h"
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

struct tensor *tensor_alloc(size_t *shape, size_t shape_size)
{
    struct tensor *t = tensor_no_grad_alloc(shape, shape_size);
    if (!t)
    {
        return NULL;
    }

    t->grad = tensor_no_grad_zero_alloc(shape, shape_size);
    if (!t->grad)
    {
        free(t);
        return NULL;
    }
    return t;
}

struct tensor* tensor_no_grad_alloc(size_t *shape, size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
       data_size *= shape[i];
    }

    struct tensor *t = (struct tensor *)malloc(sizeof(struct tensor));
    if (!t)
    {
        return NULL;
    }

    double* data = (double *)malloc(data_size * sizeof(double));
    if (!data)
    {
        free(t);
        return NULL;
    }

    size_t *_shape = (size_t *)malloc((shape_size + 1)* sizeof(size_t));
    if (!_shape)
    {
        free(t);
        free(data);
        return NULL;
    }

    // Init _shape
    memcpy(_shape, shape, shape_size * sizeof(size_t));
    _shape[shape_size] = 0;

    t->data = data;
    t->shape = _shape;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;

    return t;
}

struct tensor* tensor_no_grad_zero_alloc(size_t *shape, size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor *t = (struct tensor *)malloc(sizeof(struct tensor));
    if (!t)
    {
        return NULL;
    }

    double* data = (double *)calloc(data_size, sizeof(double));
    if (!data)
    {
        free(t);
        return NULL;
    }

    size_t *_shape = (size_t *)malloc((shape_size + 1)* sizeof(size_t));
    if (!_shape)
    {
        free(t);
        free(data);
        return NULL;
    }

    // Init _shape
    memcpy(_shape, shape, shape_size * sizeof(size_t));
    _shape[shape_size] = 0;


    t->data = data;
    t->shape = _shape;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;

    return t;
}

struct tensor *tensor2d_alloc(size_t rows, size_t cols)
{
    struct tensor *t = tensor2d_no_grad_alloc(rows, cols);
    if (!t)
    {
        return NULL;
    }

    t->grad = tensor2d_no_grad_zero_alloc(rows, cols);
    if (!t->grad)
    {
        free(t);
        return NULL;
    }
    return t;
}

struct tensor *tensor2d_no_grad_alloc(size_t rows, size_t cols)
{
    struct tensor *t = (struct tensor *)malloc(sizeof(struct tensor));
    if (!t)
    {
        return NULL;
    }

    double *data = (double *)malloc(rows * cols * sizeof(double));
    if (!data)
    {
        free(t);
        return NULL;
    }

    size_t *shape = (size_t *)malloc(3 * sizeof(size_t));
    if (!shape)
    {
        free(t);
        free(data);
        return NULL;
    }

    // Set the 2 dimensions and null terminator
    shape[0] = rows;
    shape[1] = cols;
    shape[2] = 0;

    t->data = data;
    t->shape = shape;
    t->node = NULL;
    t->data_size = rows * cols;
    t->shape_size = 2;
    t->grad = NULL;

    return t;
}

struct tensor *tensor2d_no_grad_zero_alloc(size_t rows, size_t cols)
{
    struct tensor *t = (struct tensor *)malloc(sizeof(struct tensor));
    if (!t)
    {
        return NULL;
    }

    double *data = (double *)calloc(rows * cols, sizeof(double)); // Ensure 0 for all cells
    if (!data)
    {
        free(t);
        return NULL;
    }

    size_t *shape = (size_t *)malloc(3 * sizeof(size_t));
    if (!shape)
    {
        free(t);
        free(data);
        return NULL;
    }

    // Set the 2 dimensions and null terminator
    shape[0] = rows;
    shape[1] = cols;
    shape[2] = 0;

    t->data = data;
    t->shape = shape;
    t->node = NULL;
    t->data_size = rows * cols;
    t->shape_size = 2;
    t->grad = NULL;

    return t;
}

struct tensor *tensor2d_alloc_like(struct tensor *t)
{
    if (!t->shape)
    {
        return NULL;
    }
    if (!t->shape[0] || !t->shape[1])
    {
        return NULL;
    }

    return tensor2d_alloc(t->shape[0], t->shape[1]);
}

void tensor_free(struct tensor *t)
{
    if (!t)
    {
        return;
    }

    free(t->data);
    t->data = NULL;
    
    free(t->shape);
    t->shape = NULL;

    if (t->grad)
    {
        tensor_no_grad_free(t->grad);
        t->grad = NULL;
    }

    if (t->node)
    {
        t->node = NULL;  // The node will be freed separately
    }

    free(t);
}

void tensor_no_grad_free(struct tensor *t)
{
    if (!t)
    {
        return;
    }

    free(t->data);
    t->data = NULL;

    free(t->shape);
    t->shape = NULL;
    
    free(t);
}

cgrad_error tensor2d_copy(const struct tensor *const src, struct tensor *const dest)
{
    if (!src || !dest)
    {
        return TENSOR_NULL;
    }
    if (!src->data || !dest->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (!src->shape || !dest->shape_size)
    {
        return TENSOR_SHAPE_NULL;
    }
    if (src->shape[0] != dest->shape[0] || src->shape[1] != dest->shape[1])
    {
        return TENSOR_SHAPE_MISMATCH;
    }
        
    memcpy(dest->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    return NO_ERROR;
}

cgrad_error tensor_copy(const struct tensor *const src, struct tensor *const dest)
{
    if (!src || !dest)
    {
        return TENSOR_NULL;
    }
    if (!src->data || !dest->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (!src->shape || !dest->shape_size)
    {
        return TENSOR_SHAPE_NULL;
    }
    if (src->shape_size != dest->shape_size)
    {
        return TENSOR_SHAPE_MISMATCH;
    }

    for (size_t i = 0; i < src->shape_size; i++)
    {
        if (src->shape[i] != dest->shape[i])
        {
            return TENSOR_SHAPE_MISMATCH;
        }
    }

    memcpy(dest->data, src->data, sizeof(double) * src->data_size);

    return NO_ERROR;
}


// Function to create a copy of a tensor and return a new instance
struct tensor *tensor_clone(const struct tensor *const src)
{
    if (!src)
    {
        return NULL;
    }

    struct tensor *new_tensor = tensor2d_alloc(src->shape[0], src->shape[1]);
    if (!new_tensor)
    {
        return NULL;
    }

    memcpy(new_tensor->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    return new_tensor;
}

void tensor_fill(struct tensor *const t, double value)
{
    if (!t || !t->data)
    {
        return;
    }

    size_t data_size = t->data_size;
    for (size_t i = 0; i < data_size; i++)
    {
        t->data[i] = value;
    }
}

cgrad_error tensor_add_inplace(struct tensor *A, const struct tensor *const B)
{
    if (!A || !B)
    {
        return TENSOR_NULL;
    }
    if (!A->data || !B->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (!A->shape || !B->shape)
    {
        return TENSOR_SHAPE_NULL;
    }
    if (A->data_size != B->data_size)
    {
        return TENSOR_DATA_SIZE_MISMATCH;
    }
    if (!tensor_same_shape(A, B))
    {
        return false;
    }

    tensor_add_inplace_unchecked(A, B);
    return NO_ERROR;
}

void tensor_add_inplace_unchecked(struct tensor *A, const struct tensor *const B)
{
    for (size_t i = 0; i < A->data_size; i++)
    {
        A->data[i] += B->data[i];
    }
}

bool tensor_same_shape(const struct tensor *const A, const struct tensor *const B)
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

void print_tensor(const struct tensor *const t)
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
