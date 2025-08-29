#ifndef ALLOCATORS_H
#define ALLOCATORS_H

#include "cgrad/datastructures/tensor_list.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_allocator.h"

struct cgrad_env
{
    unsigned int seed;
    struct tensor_allocator tensor_alloc;
    struct tensor_list *tensor_alloc_intermediates;
    struct computational_graph_allocator graph_alloc;
};

cgrad_error cgrad_env_init(struct cgrad_env *env, const unsigned int seed, const size_t intermediates_capacity);
void cgrad_env_cleanup(struct cgrad_env *env);
static inline cgrad_error cgrad_env_free_intermediates(struct cgrad_env *env);

static inline struct tensor *tensor_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_no_grad_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_no_grad_zero_alloc(struct cgrad_env *env, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline struct tensor *tensor_from_array_alloc(struct cgrad_env *env, const void *data, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);
static inline void tensor_free(struct cgrad_env *env, struct tensor *ptr);
static inline void tensor_no_grad_free(struct cgrad_env *env, struct tensor *ptr);
static inline struct tensor* tensor_clone(struct cgrad_env *env, struct tensor *src);

static inline cgrad_error cgrad_env_free_intermediates(struct cgrad_env *env)
{
    if (!env)
    {
        return CGRAD_ENV_NULL;
    }

    for (size_t i = 0; i < env->tensor_alloc_intermediates->size; i++)
    {
        tensor_free(env, env->tensor_alloc_intermediates->data[i]);
    }

    env->tensor_alloc_intermediates->size = 0;
}

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

static inline struct tensor* tensor_clone(struct cgrad_env *env, struct tensor *src)
{
    return tensor_allocator_clone(&env->tensor_alloc, src);
}


#endif