#include "memory/tensor_cpu_allocator.h"

struct tensor *tensor_cpu_alloc(void *pool, size_t *shape, size_t shape_size);

struct tensor* tensor_cpu_no_grad_alloc(void *pool, size_t *shape, size_t shape_size);

struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, size_t *shape, size_t shape_size);

void tensor_cpu_free(void *pool, struct tensor *t);

void tensor_cpu_no_grad_free(void *pool, struct tensor *t);

struct tensor *tensor2d_cpu_alloc(void *pool, size_t rows, size_t cols);

struct tensor *tensor2d_cpu_no_grad_alloc(void *pool, size_t rows, size_t cols);

struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src);


struct tensor_allocator make_tensor_cpu_allocator(struct tensor_pool *pool)
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

struct tensor *tensor_cpu_alloc(void *pool, size_t *shape, size_t shape_size)
{
    return tensor_pool_alloc((struct tensor_pool *)pool, shape, shape_size);
}

struct tensor* tensor_cpu_no_grad_alloc(void *pool, size_t *shape, size_t shape_size)
{
    return tensor_pool_no_grad_alloc((struct tensor_pool *)pool, shape, shape_size);
}

struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, size_t *shape, size_t shape_size)
{
    return tensor_pool_no_grad_zero_alloc((struct tensor_pool *)pool, shape, shape_size);
}

void tensor_cpu_free(void *pool, struct tensor *t)
{
    return tensor_pool_free((struct tensor_pool *)pool, t);
}

void tensor_cpu_no_grad_free(void *pool, struct tensor *t)
{
    return tensor_pool_no_grad_free((struct tensor_pool *)pool, t);
}

struct tensor *tensor2d_cpu_alloc(void *pool, size_t rows, size_t cols)
{
    return tensor2d_pool_alloc((struct tensor_pool *)pool, rows, cols);
}

struct tensor *tensor2d_cpu_no_grad_alloc(void *pool, size_t rows, size_t cols)
{
    return tensor2d_pool_no_grad_alloc((struct tensor_pool *)pool, rows, cols);
}

struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src)
{
    return tensor_pool_clone((struct tensor_pool *)pool, src);
}