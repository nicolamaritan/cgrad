#ifndef TENSOR_ALLOC_H
#define TENSOR_ALLOC_H

#include "cgrad/cgrad_env.h"
#include "cgrad/memory/tensor/tensor_allocator.h"

static inline struct tensor *tensor_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_no_grad_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_no_grad_zero_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_from_array_alloc(struct cgrad_env *env, const void *data, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline void tensor_free(struct cgrad_env *env, struct tensor *ptr);
static inline void tensor_no_grad_free(struct cgrad_env *env, struct tensor *ptr);
static inline struct tensor *tensor_clone(struct cgrad_env *env, struct tensor *src);

static inline struct tensor *tensor_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return tensor_allocator_alloc(&env->tensor_alloc, shape, shape_size, dtype);
}

static inline struct tensor *tensor_no_grad_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return tensor_allocator_no_grad_alloc(&env->tensor_alloc, shape, shape_size, dtype);
}

static inline struct tensor *tensor_no_grad_zero_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return tensor_allocator_no_grad_zero_alloc(&env->tensor_alloc, shape, shape_size, dtype);
}

static inline struct tensor *tensor_from_array_alloc(struct cgrad_env *env, const void *data, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype)
{
    return tensor_allocator_from_array_alloc(&env->tensor_alloc, data, shape, shape_size, dtype);
}

static inline void tensor_free(struct cgrad_env *env, struct tensor *ptr)
{
    tensor_allocator_free(&env->tensor_alloc, ptr);
}

static inline void tensor_no_grad_free(struct cgrad_env *env, struct tensor *ptr)
{
    tensor_allocator_no_grad_free(&env->tensor_alloc, ptr);
}

static inline struct tensor *tensor_clone(struct cgrad_env *env, struct tensor *src)
{
    return tensor_allocator_clone(&env->tensor_alloc, src);
}

#endif