#ifndef TENSOR_H
#define TENSOR_H

#include "cgrad/dtypes.h"
#include "cgrad/config.h"
#include "cgrad/error.h"
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
    void *data;                          /**< Pointer to the data stored in the tensor. */
    cgrad_dtype dtype;                   /**< Data type pointed by data */
    size_t shape[TENSOR_MAX_SHAPE_SIZE]; /**< Shape of the tensor. */
    size_t stride[TENSOR_MAX_SHAPE_SIZE];
    size_t data_size;                      /**< Total number of elements in the tensor. */
    size_t shape_size;                     /**< Number of dimensions in the tensor. */
    struct computational_graph_node *node; /**< Pointer to the computational graph node for gradient tracking. */
    struct tensor *grad;                   /**< Pointer to the gradient tensor. */
};

#endif