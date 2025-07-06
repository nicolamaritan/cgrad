#include "memory/tensor_pool_alloc.h"
#include <string.h>
#include <stdlib.h>

struct tensor *tensor_pool_alloc(struct tensor_pool *pool, size_t *shape, size_t shape_size)
{
    struct tensor *t = tensor_pool_no_grad_alloc(pool, shape, shape_size);
    if (!t)
    {
        return NULL;
    }

    t->grad = tensor_pool_no_grad_zero_alloc(pool, shape, shape_size);
    if (!t->grad)
    {
        tensor_pool_free(pool, t);
        return NULL;
    }
    return t;
}

struct tensor *tensor_pool_no_grad_alloc(struct tensor_pool *pool, size_t *shape, size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor *t = tensor_pool_tensor_alloc(pool);
    if (!t)
    {
        return NULL;
    }

    double *data = (double *)tensor_pool_data_alloc(pool);
    if (!data)
    {
        tensor_pool_tensor_free(pool, t);
        return NULL;
    }

    size_t *_shape = (size_t *)malloc((shape_size + 1) * sizeof(size_t));
    if (!_shape)
    {
        tensor_pool_tensor_free(pool, t);
        tensor_pool_data_free(pool, data);
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

struct tensor *tensor_pool_no_grad_zero_alloc(struct tensor_pool *pool, size_t *shape, size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor *t = tensor_pool_tensor_alloc(pool);
    if (!t)
    {
        return NULL;
    }

    double *data = (double *)tensor_pool_data_zero_alloc(pool);
    if (!data)
    {
        tensor_pool_tensor_free(pool, t);
        return NULL;
    }

    size_t *_shape = (size_t *)malloc((shape_size + 1) * sizeof(size_t));
    if (!_shape)
    {
        tensor_pool_tensor_free(pool, t);
        tensor_pool_data_free(pool, data);
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

/**
 * @brief Frees the memory allocated for the tensor, including its data and shape.
 *
 * @param pool
 * @param t Pointer to the tensor to be freed.
 */
void tensor_pool_free(struct tensor_pool *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    tensor_pool_data_free(pool, t->data);
    t->data = NULL;

    free(t->shape);
    t->shape = NULL;

    if (t->grad)
    {
        tensor_pool_no_grad_free(pool, t->grad);
        t->grad = NULL;
    }

    if (t->node)
    {
        t->node = NULL; // The node will be freed separately
    }

    tensor_pool_tensor_free(pool, t);
}

void tensor_pool_no_grad_free(struct tensor_pool *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    tensor_pool_data_free(pool, t->data);
    t->data = NULL;

    free(t->shape);
    t->shape = NULL;

    tensor_pool_tensor_free(pool, t);
}

struct tensor *tensor2d_pool_alloc(struct tensor_pool *pool, size_t rows, size_t cols)
{
    struct tensor *t = tensor2d_pool_no_grad_alloc(pool, rows, cols);
    if (!t)
    {
        return NULL;
    }

    t->grad = tensor2d_pool_no_grad_alloc(pool, rows, cols);
    if (!t->grad)
    {
        tensor_pool_free(pool, t);
        return NULL;
    }
    return t;
}

struct tensor *tensor2d_pool_no_grad_alloc(struct tensor_pool *pool, size_t rows, size_t cols)
{
    struct tensor *t = tensor_pool_tensor_alloc(pool); 
    if (!t)
    {
        return NULL;
    }

    double *data = (double *)tensor_pool_data_zero_alloc(pool);
    if (!data)
    {
        tensor_pool_tensor_free(pool, t);
        return NULL;
    }

    size_t *shape = (size_t *)malloc(3 * sizeof(size_t));
    if (!shape)
    {
        tensor_pool_tensor_free(pool, t);
        tensor_pool_data_free(pool, data);
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

struct tensor *tensor_pool_clone(struct tensor_pool *pool, const struct tensor *const src)
{
    if (!src)
    {
        return NULL;
    }

    struct tensor *new_tensor = tensor2d_pool_alloc(pool, src->shape[0], src->shape[1]);
    if (!new_tensor)
    {
        return NULL;
    }

    memcpy(new_tensor->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    return new_tensor;
}