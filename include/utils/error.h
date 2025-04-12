#ifndef ERROR_H
#define ERROR_H

/**
 * @enum cgrad_error
 * @brief Enumeration of possible error codes.
 */
typedef enum
{
    NO_ERROR = 0,                /**< Operation was successful. */

    // Tensor operations errors
    TENSOR_NULL,                 /**< Tensor pointer is null. */
    TENSOR_SHAPE_NULL,           /**< Tensor shape pointer is null. */
    TENSOR_WRONG_SHAPE,          /**< Tensor has an incorrect shape. */
    TENSOR_DATA_NULL,            /**< Tensor data pointer is null. */
    TENSOR_INDEX_OUT_OF_BOUNDS,  /**< Index is out of bounds for the tensor. */
    TENSOR_SHAPE_MISMATCH,       /**< Shapes of tensors do not match. */
    TENSOR_DATA_SIZE_MISMATCH,    /**< Data sizes of tensors do not match. */

    // Model errors
    MODEL_MAX_PARAMS_EXCEEDED,

    // Autograd errors
    AUTOGRAD_MAX_PARENTS_EXCEEDED,
    AUTOGRAD_MAX_CHILDREN_EXCEEDED,
    AUTOGRAD_MAX_TARGETS_EXCEEDED,
} cgrad_error;

#endif