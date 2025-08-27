#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include <string.h>

static struct tensor *tensor_cpu_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

static struct tensor *tensor_cpu_no_grad_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

static struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

static struct tensor *tensor_cpu_from_array_alloc(void *pool, const void *data, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

static void tensor_cpu_free(void *pool, struct tensor *t);

static void tensor_cpu_no_grad_free(void *pool, struct tensor *t);

static struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src);

static void compute_stride(size_t *const shape, size_t *const stride, size_t const shape_size);

cgrad_error tensor_cpu_allocator_init(struct tensor_allocator *const tensor_alloc)
{
    if (!tensor_alloc)
    {
        return TENSOR_ALLOCATOR_NULL;
    }

    struct tensor_cpu_pool *tensor_pool = calloc(1, sizeof(struct tensor_cpu_pool));
    if (!tensor_pool)
    {
        return TENSOR_POOL_ALLOCATION_FAILED;
    }

    cgrad_error err = tensor_cpu_pool_init(tensor_pool);
    if (err != NO_ERROR)
    {
        return err;
    }

    tensor_alloc->alloc = tensor_cpu_alloc,
    tensor_alloc->no_grad_alloc = tensor_cpu_no_grad_alloc,
    tensor_alloc->no_grad_zero_alloc = tensor_cpu_no_grad_zero_alloc,
    tensor_alloc->from_array_alloc = tensor_cpu_from_array_alloc;
    tensor_alloc->free = tensor_cpu_free,
    tensor_alloc->no_grad_free = tensor_cpu_no_grad_free,
    tensor_alloc->clone = tensor_cpu_clone,
    tensor_alloc->pool = tensor_pool;

    return NO_ERROR;
}

void tensor_cpu_allocator_cleanup(struct tensor_allocator *const tensor_alloc)
{
    if (!tensor_alloc)
    {
        return;
    }

    tensor_cpu_pool_cleanup(tensor_alloc->pool);
    free(tensor_alloc->pool);
}

static struct tensor *tensor_cpu_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_cpu_no_grad_alloc(cpu_pool, shape, shape_size, dtype);
    if (!t)
    {
        return NULL;
    }

    // Allocate gradient only for real value tensors
    if (dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64)
    {
        t->grad = tensor_cpu_no_grad_zero_alloc(cpu_pool, shape, shape_size, dtype);
        if (!t->grad)
        {
            tensor_cpu_free(cpu_pool, t);
            return NULL;
        }
    }
    else
    {
        t->grad = NULL;
    }
    return t;
}

static struct tensor *tensor_cpu_no_grad_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_cpu_pool_tensor_alloc(cpu_pool, data_size);
    if (!t)
    {
        return NULL;
    }

    void *data = tensor_cpu_pool_data_zero_alloc(cpu_pool, data_size * dtype_sizeof(dtype));
    if (!data)
    {
        tensor_cpu_pool_tensor_free(cpu_pool, t);
        return NULL;
    }

    // Init _shape
    memcpy(t->shape, shape, shape_size * sizeof(size_t));

    compute_stride(t->shape, t->stride, shape_size);

    t->data = data;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;
    t->dtype = dtype;

    return t;
}

static struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_cpu_pool_tensor_alloc(cpu_pool, data_size);
    if (!t)
    {
        return NULL;
    }

    void *data = tensor_cpu_pool_data_zero_alloc(cpu_pool, data_size * dtype_sizeof(dtype));
    if (!data)
    {
        tensor_cpu_pool_tensor_free(cpu_pool, t);
        return NULL;
    }

    // Init _shape
    memcpy(t->shape, shape, shape_size * sizeof(size_t));

    compute_stride(t->shape, t->stride, shape_size);

    t->data = data;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;
    t->dtype = dtype;

    return t;
}

static struct tensor *tensor_cpu_from_array_alloc(void *pool, const void *data, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    struct tensor *t = tensor_cpu_alloc(pool, shape, shape_size, dtype);
    if (!t)
    {
        return NULL;
    }

    memcpy(t->data, data, t->data_size * dtype_sizeof(dtype));

    return t;
}

static void tensor_cpu_free(void *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    tensor_cpu_pool_data_free(cpu_pool, t->data);
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

    tensor_cpu_pool_tensor_free(cpu_pool, t);
}

static void tensor_cpu_no_grad_free(void *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    tensor_cpu_pool_data_free(cpu_pool, t->data);
    t->data = NULL;

    tensor_cpu_pool_tensor_free(cpu_pool, t);
}

static struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src)
{
    if (!src)
    {
        return NULL;
    }

    struct tensor *new_tensor = tensor_cpu_alloc(pool, src->shape, src->shape_size, src->dtype);
    if (!new_tensor)
    {
        return NULL;
    }

    memcpy(new_tensor->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    return new_tensor;
}

static void compute_stride(size_t *const shape, size_t *const stride, size_t const shape_size)
{
    stride[shape_size - 1] = 1;
    // Use int for allowing i = 0
    for (int i = shape_size - 2; i >= 0; i--)
    {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}