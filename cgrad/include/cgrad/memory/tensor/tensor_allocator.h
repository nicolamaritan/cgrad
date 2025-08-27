#ifndef TENSOR_ALLOCATOR_H
#define TENSOR_ALLOCATOR_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/dtypes.h"
#include "cgrad/error.h"
#include <stddef.h>

typedef struct tensor *(*alloc_fn)(void*, const size_t *const, const size_t, const cgrad_dtype);
typedef struct tensor *(*from_array_alloc_fn)(void*, const void*, const size_t *const, const size_t, const cgrad_dtype);
typedef void (*free_fn)(void*, struct tensor*);
typedef struct tensor *(*clone_fn)(void*, const struct tensor *const);

struct tensor_allocator
{
    alloc_fn alloc;
    alloc_fn no_grad_alloc;
    alloc_fn no_grad_zero_alloc;
    from_array_alloc_fn from_array_alloc;
    free_fn free;
    free_fn no_grad_free;
    clone_fn clone;
    void *pool;
};

static inline struct tensor *tensor_allocator_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_allocator_no_grad_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_allocator_no_grad_zero_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_allocator_from_array_alloc(struct tensor_allocator *allocator, const void *data, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline void tensor_allocator_free(struct tensor_allocator *allocator, struct tensor *ptr);
static inline void tensor_allocator_no_grad_free(struct tensor_allocator *allocator, struct tensor *ptr);
static inline struct tensor* tensor_allocator_clone(struct tensor_allocator *allocator, struct tensor *src);

static inline struct tensor *tensor_allocator_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return allocator->alloc(allocator->pool, shape, shape_size, dtype);
}

static inline struct tensor *tensor_allocator_no_grad_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return allocator->no_grad_alloc(allocator->pool, shape, shape_size, dtype);
}

static inline struct tensor *tensor_allocator_no_grad_zero_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return allocator->no_grad_zero_alloc(allocator->pool, shape, shape_size, dtype);
}

static inline struct tensor *tensor_allocator_from_array_alloc(struct tensor_allocator *allocator, const void *data, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return allocator->from_array_alloc(allocator->pool, data, shape, shape_size, dtype);
}

static inline void tensor_allocator_free(struct tensor_allocator *allocator, struct tensor *ptr)
{
    allocator->free(allocator->pool, ptr);
}

static inline void tensor_allocator_no_grad_free(struct tensor_allocator *allocator, struct tensor *ptr)
{
    allocator->no_grad_free(allocator->pool, ptr);
}

static inline struct tensor* tensor_allocator_clone(struct tensor_allocator *allocator, struct tensor *src)
{
    return allocator->clone(allocator->pool, src);
}

#endif