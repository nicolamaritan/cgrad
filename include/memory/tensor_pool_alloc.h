#ifndef TENSOR_ALLOC_H
#define TENSOR_ALLOC_H

#include "memory/tensor_pool.h"

/**
 * @brief Allocates a new tensor with the specified shape and gradient tracking.
 *
 * @param pool
 * @param shape Pointer to the array containing the shape of the tensor.
 * @param shape_size Number of dimensions in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor_pool_alloc(struct tensor_pool *pool, size_t *shape, size_t shape_size);

/**
 * @brief Allocates a new tensor with the specified shape without gradient tracking.
 *
 * @param pool
 * @param shape Pointer to the array containing the shape of the tensor.
 * @param shape_size Number of dimensions in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor* tensor_pool_no_grad_alloc(struct tensor_pool *pool, size_t *shape, size_t shape_size);

struct tensor *tensor_pool_no_grad_zero_alloc(struct tensor_pool *pool, size_t *shape, size_t shape_size);

/**
 * @brief Frees the memory allocated for the tensor, including its data and shape.
 *
 * @param pool
 * @param t Pointer to the tensor to be freed.
 */
void tensor_pool_free(struct tensor_pool *pool, struct tensor *t);


void tensor_pool_no_grad_free(struct tensor_pool *pool, struct tensor *t);

/**
 * @brief Allocates a new 2D tensor with the specified number of rows and columns and gradient tracking.
 *
 * @param pool
 * @param rows Number of rows in the tensor.
 * @param cols Number of columns in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor2d_pool_alloc(struct tensor_pool *pool, size_t rows, size_t cols);

/**
 * @brief Allocates a new 2D tensor with the specified number of rows and columns without gradient tracking.
 *
 * @param pool
 * @param rows Number of rows in the tensor.
 * @param cols Number of columns in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor2d_pool_no_grad_alloc(struct tensor_pool *pool, size_t rows, size_t cols);

/**
 * @brief Creates a clone of the given tensor.
 *
 * @param src Pointer to the tensor to be cloned.
 * @return Pointer to the cloned tensor, or NULL if cloning failed.
 */
struct tensor *tensor_pool_clone(struct tensor_pool *pool, const struct tensor *const src);

#endif