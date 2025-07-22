#ifndef TENSOR_H
#define TENSOR_H

#include "tensor/dtypes.h"
#include "config.h"
#include "utils/error.h"
#include <stddef.h>
#include <stdbool.h>

struct computational_graph_node;
struct tensor;

/**
 * @struct tensor
 * @brief Represents a tensor with optional gradient tracking.
 *
 * This structure holds the data and shape of the tensor, as well as a pointer to a
 * computational graph node for gradient tracking.
 */
struct tensor
{
    void *data;        /**< Pointer to the data stored in the tensor. */
    dtype dtype;
    size_t shape[TENSOR_MAX_SHAPE_SIZE];       /**< Shape of the tensor. */
    size_t data_size;    /**< Total number of elements in the tensor. */
    size_t shape_size;   /**< Number of dimensions in the tensor. */
    struct computational_graph_node *node; /**< Pointer to the computational graph node for gradient tracking. */
    struct tensor *grad;        /**< Pointer to the gradient tensor. */
};


// Tensor allocation

/**
 * @brief Allocates a new tensor with the specified shape and gradient tracking.
 *
 * @param shape Pointer to the array containing the shape of the tensor.
 * @param shape_size Number of dimensions in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor_alloc(size_t *shape, size_t shape_size);

/**
 * @brief Allocates a new tensor with the specified shape without gradient tracking.
 *
 * @param shape Pointer to the array containing the shape of the tensor.
 * @param shape_size Number of dimensions in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor* tensor_no_grad_alloc(size_t *shape, size_t shape_size);

/**
 * @brief Allocates a new tensor with the specified shape, initialized to zero, without gradient tracking.
 *
 * @param shape Pointer to the array containing the shape of the tensor.
 * @param shape_size Number of dimensions in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor* tensor_no_grad_zero_alloc(size_t *shape, size_t shape_size);

/**
 * @brief Allocates a new 2D tensor with the specified number of rows and columns and gradient tracking.
 *
 * @param rows Number of rows in the tensor.
 * @param cols Number of columns in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor2d_alloc(size_t rows, size_t cols);

/**
 * @brief Allocates a new 2D tensor with the specified number of rows and columns without gradient tracking.
 *
 * @param rows Number of rows in the tensor.
 * @param cols Number of columns in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor2d_no_grad_alloc(size_t rows, size_t cols);

/**
 * @brief Allocates a new 2D tensor with the specified number of rows and columns, initialized to zero, without gradient tracking.
 *
 * @param rows Number of rows in the tensor.
 * @param cols Number of columns in the tensor.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor2d_no_grad_zero_alloc(size_t rows, size_t cols);

/**
 * @brief Allocates a new 2D tensor with the same shape as the given tensor.
 *
 * @param t Pointer to the tensor whose shape will be copied.
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 */
struct tensor *tensor2d_alloc_like(struct tensor *t);

/**
 * @brief Frees the memory allocated for the tensor, including its data and shape.
 *
 * @param t Pointer to the tensor to be freed.
 */
void tensor_free(struct tensor *t);

/**
 * @brief Frees the memory allocated for the tensor without gradient tracking.
 *
 * @param t Pointer to the tensor to be freed.
 */
void tensor_no_grad_free(struct tensor *t);

// Non-differentiable tensor operations

/**
 * @brief Creates a clone of the given tensor.
 *
 * @param src Pointer to the tensor to be cloned.
 * @return Pointer to the cloned tensor, or NULL if cloning failed.
 */
struct tensor *tensor_clone(const struct tensor *const src);

/**
 * @brief Copies the contents of a 2D tensor from source to destination with bounds checking.
 *
 * @param src Pointer to the source tensor.
 * @param dest Pointer to the destination tensor.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error tensor2d_copy(const struct tensor *const src, struct tensor *const dest);

/**
 * @brief Copies the contents of a tensor from source to destination with bounds checking.
 *
 * @param src Pointer to the source tensor.
 * @param dest Pointer to the destination tensor.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error tensor_copy(const struct tensor *const src, struct tensor *const dest);

// Helper functions
/**
 * @brief Checks if two tensors have the same shape.
 *
 * @param A Pointer to the first tensor.
 * @param B Pointer to the second tensor.
 * @return True if the tensors have the same shape, otherwise false.
 */
bool tensor_same_shape(const struct tensor *const A, const struct tensor *const B);

// Debug
/**
 * @brief Prints the contents of the tensor.
 *
 * @param t Pointer to the tensor to be printed.
 */
void print_tensor(const struct tensor *const t);

// Inline definitions
/**
 * @brief TODO add documentation
 */
static inline cgrad_error tensor_check_null(const struct tensor *const t)
{
    if (t == NULL)
    {
        return TENSOR_NULL;
    }
    if (t->data == NULL)
    {
        return TENSOR_DATA_NULL;
    }
    return NO_ERROR;
}

#endif