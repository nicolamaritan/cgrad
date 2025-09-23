#ifndef LINEAR_H
#define LINEAR_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/datastructures/tensor_list.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/cgrad_env.h"
#include <stddef.h>

/**
 * @struct linear
 * @brief Represents a fully connected (linear) layer in a neural network.
 *
 * This structure models a linear (affine) layer, which performs the transformation:
 * \f$ y = xW^T + b \f$
 * where:
 *   - `x` is the input,
 *   - `W` is the weight matrix,
 *   - `b` is the bias vector.
 *
 * Tensors are managed via pointers to `struct tensor`.
 */
struct linear
{
    struct tensor *weight;
    struct tensor *bias;
    size_t in_dim;
    size_t out_dim;
    struct cgrad_env *env;
};

/**
 * @brief Initializes a linear (fully connected) layer.
 *
 * This function allocates the weight and bias tensors
 * for a linear layer using the provided CGRAD environment. The weight
 * tensor has shape `[in_dim, out_dim]`, and the bias tensor has shape
 * `[1, out_dim]`. If any allocation fails, allocated resources are freed
 * and an error code is returned.
 *
 * @param layer Pointer to the `linear` layer structure to initialize.
 *        Must not be NULL.
 * @param in_dim Input dimension size (number of features).
 * @param out_dim Output dimension size (number of units).
 * @param dtype Data type of the tensors
 * @param env Pointer to the CGRAD environment used for memory allocation.
 *        Must not be NULL.
 *
 * @return
 * - `NO_ERROR` if the layer was successfully initialized.
 * - `LINEAR_NULL` if `layer` is NULL.
 * - `CGRAD_ENV_NULL` if `env` is NULL.
 * - `TENSOR_ALLOCATION_FAILED` if allocation of weight or bias fails.
 *
 * @note On allocation failure, this function frees partially allocated
 *       resources before returning.
 */
cgrad_error linear_init(struct linear *const layer, const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct cgrad_env *const env);
cgrad_error linear_forward(struct linear *const layer, struct tensor *const x, struct tensor **const out, const bool track_grad);

/**
 * @brief Initializes the weights of a linear layer using Xavier initialization.
 *
 * This function applies Xavier (Glorot) initialization to the weight tensor
 * of a given linear layer. The initialization method depends on the tensor's
 * data type:
 * - `DTYPE_FLOAT64`: calls `linear_xavier_init_f64()`.
 * - `DTYPE_FLOAT32`: calls `linear_xavier_init_f32()`.
 *
 * @param layer Pointer to the `linear` layer to initialize. Must not be NULL.
 *
 * @return
 * - `NO_ERROR` if the weights were successfully initialized.
 * - `LINEAR_NULL` if `layer` is NULL.
 * - `LINEAR_INVALID_DTYPE` if the layer's weight tensor has an unsupported dtype.
 */
cgrad_error linear_xavier_init(struct linear *const layer);

/**
 * @brief Cleans up a linear layer by freeing its allocated tensors.
 *
 * This function releases the memory for the weight and bias tensors of
 * a linear layer using the tensor allocator from the associated CGRAD
 * environment. It does not free the `linear` structure itself.
 *
 * @param layer Pointer to the `linear` layer to clean up. If NULL,
 *        the function does nothing.
 *
 * @note After calling this function, the weight and bias pointers in
 *       the layer are no longer valid.
 */
void linear_cleanup(struct linear *const layer);

#endif