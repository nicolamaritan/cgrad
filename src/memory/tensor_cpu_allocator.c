#include "memory/tensor_cpu_allocator.h"
#include <string.h>

struct tensor *tensor_cpu_alloc(void *pool, const size_t *const shape, const size_t shape_size);

struct tensor* tensor_cpu_no_grad_alloc(void *pool, const size_t *const shape, const size_t shape_size);

struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, const size_t *const shape, const size_t shape_size);

void tensor_cpu_free(void *pool, struct tensor *t);

void tensor_cpu_no_grad_free(void *pool, struct tensor *t);

struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src);

struct tensor_allocator make_tensor_cpu_allocator(struct tensor_cpu_pool *pool)
{
    return (struct tensor_allocator)
    {
        .alloc = tensor_cpu_alloc,
        .no_grad_alloc = tensor_cpu_no_grad_alloc,
        .no_grad_zero_alloc = tensor_cpu_no_grad_zero_alloc,
        .free = tensor_cpu_free,
        .no_grad_free = tensor_cpu_no_grad_free,
        .clone = tensor_cpu_clone,
        .pool = pool
    };    
}

struct tensor *tensor_cpu_alloc(void *pool, const size_t *const shape, const size_t shape_size)
{
    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_cpu_no_grad_alloc(cpu_pool, shape, shape_size);
    if (!t)
    {
        return NULL;
    }

    t->grad = tensor_cpu_no_grad_zero_alloc(cpu_pool, shape, shape_size);
    if (!t->grad)
    {
        tensor_cpu_free(cpu_pool, t);
        return NULL;
    }
    return t;
}

struct tensor* tensor_cpu_no_grad_alloc(void *pool, const size_t *const shape, const size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_pool_tensor_alloc(cpu_pool, data_size);
    if (!t)
    {
        return NULL;
    }

    double *data = (double *)tensor_pool_data_alloc(cpu_pool, data_size);
    if (!data)
    {
        tensor_pool_tensor_free(cpu_pool, t);
        return NULL;
    }

    // Init _shape
    memcpy(t->shape, shape, shape_size * sizeof(size_t));

    t->data = data;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;

    return t;
}

struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, const size_t *const shape, const size_t shape_size)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_pool_tensor_alloc(cpu_pool, data_size);
    if (!t)
    {
        return NULL;
    }

    double *data = (double *)tensor_pool_data_zero_alloc(cpu_pool, data_size);
    if (!data)
    {
        tensor_pool_tensor_free(cpu_pool, t);
        return NULL;
    }

    // Init _shape
    memcpy(t->shape, shape, shape_size * sizeof(size_t));

    t->data = data;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;

    return t;
}

void tensor_cpu_free(void *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    tensor_pool_data_free(cpu_pool, t->data);
    t->data = NULL;

    if (t->grad)
    {
        tensor_cpu_no_grad_free(cpu_pool, t->grad);
        t->grad = NULL;
    }

    if (t->node)
    {
        t->node = NULL; // The node will be freed separately
    }

    tensor_pool_tensor_free(cpu_pool, t);
}

void tensor_cpu_no_grad_free(void *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    tensor_pool_data_free(cpu_pool, t->data);
    t->data = NULL;

    tensor_pool_tensor_free(cpu_pool, t);
}

struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src)
{
    if (!src)
    {
        return NULL;
    }

    struct tensor *new_tensor = tensor_cpu_alloc(pool, src->shape, src->shape_size);
    if (!new_tensor)
    {
        return NULL;
    }

    memcpy(new_tensor->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    return new_tensor;
}