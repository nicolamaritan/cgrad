#ifndef TENSOR_ALLOCATOR_H
#define TENSOR_ALLOCATOR_H

#include "tensor/tensor.h"
#include "utils/error.h"
#include <stddef.h>

typedef struct tensor *(*alloc_fn)(void*, size_t*, size_t);
typedef void (*free_fn)(void*, struct tensor*);
typedef struct tensor *(*clone_fn)(void*, const struct tensor *const);

struct tensor_allocator
{
    alloc_fn alloc;
    alloc_fn no_grad_alloc;
    alloc_fn no_grad_zero_alloc;
    free_fn free;
    free_fn no_grad_free;
    clone_fn clone;
    void *pool;
};

static inline struct tensor *tensor_allocator_alloc(struct tensor_allocator *allocator, size_t *shape, size_t shape_size);
static inline struct tensor *tensor_allocator_no_grad_alloc(struct tensor_allocator *allocator, size_t *shape, size_t shape_size);
static inline struct tensor *tensor_allocator_no_grad_zero_alloc(struct tensor_allocator *allocator, size_t *shape, size_t shape_size);
static inline void tensor_allocator_free(struct tensor_allocator *allocator, struct tensor *ptr);
static inline void tensor_allocator_no_grad_free(struct tensor_allocator *allocator, struct tensor *ptr);

struct tensor *tensor_allocator_alloc(struct tensor_allocator *allocator, size_t *shape, size_t shape_size)
{
    return allocator->alloc(allocator->pool, shape, shape_size);
}

static inline struct tensor *tensor_allocator_no_grad_alloc(struct tensor_allocator *allocator, size_t *shape, size_t shape_size)
{
    return allocator->no_grad_alloc(allocator->pool, shape, shape_size);
}

struct tensor *tensor_allocator_no_grad_zero_alloc(struct tensor_allocator *allocator, size_t *shape, size_t shape_size)
{
    return allocator->no_grad_zero_alloc(allocator->pool, shape, shape_size);
}

void tensor_allocator_free(struct tensor_allocator *allocator, struct tensor *ptr)
{
    allocator->free(allocator->pool, ptr);
}

void tensor_allocator_no_grad_free(struct tensor_allocator *allocator, struct tensor *ptr)
{
    allocator->no_grad_free(allocator->pool, ptr);
}

#endif