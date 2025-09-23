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

/**
 * @brief Allocates a new tensor using a tensor_allocator.
 *
 * This function invokes the allocation function defined in the
 * `struct tensor_allocator`, passing the memory pool, shape,
 * and data type of the tensor to be created.
 *
 * @param allocator Pointer to the `tensor_allocator` that manages allocation.
 * @param shape Array of dimensions defining the tensor's shape.
 * @param shape_size Number of dimensions specified in the `shape` array.
 * @param dtype Data type of the tensor 
 *
 * @return Pointer to the newly allocated `tensor` object, or NULL on failure.
 */
static inline struct tensor *tensor_allocator_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Allocates a tensor without gradient tracking.
 *
 * This function uses the allocator's `no_grad_alloc` to create a tensor
 * that is excluded from gradient computations.
 *
 * @param allocator Pointer to the `tensor_allocator` that manages allocation.
 * @param shape Array of dimensions defining the tensor's shape.
 * @param shape_size Number of dimensions specified in the `shape` array.
 * @param dtype Data type of the tensor
 *
 * @return Pointer to the newly allocated `tensor` object, or NULL on failure.
 */
static inline struct tensor *tensor_allocator_no_grad_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Allocates a zero-initialized tensor without gradient tracking.
 *
 * This function uses the allocator's `no_grad_zero_alloc` to create a tensor
 * initialized to zero values, excluded from gradient computations.
 *
 * @param allocator Pointer to the `tensor_allocator` that manages allocation.
 * @param shape Array of dimensions defining the tensor's shape.
 * @param shape_size Number of dimensions specified in the `shape` array.
 * @param dtype Data type of the tensor.
 *
 * @return Pointer to the newly allocated `tensor` object, or NULL on failure.
 */
static inline struct tensor *tensor_allocator_no_grad_zero_alloc(struct tensor_allocator *allocator, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Allocates a tensor from an existing data array.
 *
 * This function uses the allocator's `from_array_alloc` to wrap existing
 * data into a tensor object.
 *
 * @param allocator Pointer to the `tensor_allocator` that manages allocation.
 * @param data Pointer to the raw data buffer.
 * @param shape Array of dimensions defining the tensor's shape.
 * @param shape_size Number of dimensions specified in the `shape` array.
 * @param dtype Data type of the tensor.
 *
 * @return Pointer to the newly created `tensor` object, or NULL on failure.
 */
static inline struct tensor *tensor_allocator_from_array_alloc(struct tensor_allocator *allocator, const void *data, const size_t *shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Frees a tensor previously allocated with gradient tracking.
 *
 * This function uses the allocator's `free` callback to release memory.
 *
 * @param allocator Pointer to the `tensor_allocator` that manages memory.
 * @param ptr Pointer to the `tensor` to free.
 */
static inline void tensor_allocator_free(struct tensor_allocator *allocator, struct tensor *ptr);

/**
 * @brief Frees a tensor previously allocated without gradient tracking.
 *
 * This function uses the allocator's `no_grad_free` callback to release memory.
 *
 * @param allocator Pointer to the `tensor_allocator` that manages memory.
 * @param ptr Pointer to the `tensor` to free.
 */
static inline void tensor_allocator_no_grad_free(struct tensor_allocator *allocator, struct tensor *ptr);


/**
 * @brief Clones an existing tensor.
 *
 * This function uses the allocator's `clone` to create a copy of
 * the given tensor, including its data and metadata.
 *
 * @param allocator Pointer to the `tensor_allocator` that manages allocation.
 * @param src Pointer to the source `tensor` to clone.
 *
 * @return Pointer to the newly cloned `tensor` object, or NULL on failure.
 */
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