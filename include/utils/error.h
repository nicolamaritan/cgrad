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
    TENSOR_WRONG_SHAPE,          /**< Tensor has an incorrect shape. */
    TENSOR_DATA_NULL,            /**< Tensor data pointer is null. */
    TENSOR_INDEX_OUT_OF_BOUNDS,  /**< Index is out of bounds for the tensor. */
    TENSOR_SHAPE_MISMATCH,       /**< Shapes of tensors do not match. */
    TENSOR_DATA_SIZE_MISMATCH,    /**< Data sizes of tensors do not match. */

    // Model errors
    MODEL_MAX_PARAMS_EXCEEDED,
    MODEL_PARAMS_NULL,

    // Optimizers
    OPTIMIZER_NULL,

    // Allocator
    TENSOR_ALLOCATOR_NULL,
    COMPUTATIONAL_GRAPH_ALLOCATOR_NULL,

    // Autograd errors
    AUTOGRAD_MAX_PARENTS_EXCEEDED,
    AUTOGRAD_MAX_CHILDREN_EXCEEDED,
    AUTOGRAD_MAX_TARGETS_EXCEEDED,
    AUTOGRAD_INVALID_CONTEXT_ID,
    AUTOGRAD_CONTEXT_ID_ALREADY_TAKEN,
    AUTOGRAD_COMPUTATIONAL_GRAPH_NODE_ALLOCATION_ERROR,
    AUTOGRAD_BACKPROPAGATION_CONTEXT_NULL,
    AUTOGRAD_ALLOCATORS_NULL,
    AUTOGRAD_BACKPROPAGATION_FUNCTION_NULL,

    // Dataset
    DATASET_NULL,
    DATASET_FILE_ERROR,
    CSV_DATASET_FORMAT_ERROR,
    CSV_DATASET_DATA_NULL,

    // Permutation
    INDEXES_PERMUTATION_NULL,

    // Index Batch
    INDEXES_BATCH_NULL,

    // Memory
    MEMORY_POOL_NULL,
    MEMORY_POOL_CHUNK_ALLOCATION_FAILED,

    // General
    INPUT_PTR_NULL,
    OUTPUT_PTR_NULL,
    INVALID_BATCH_SIZE,

} cgrad_error;

#endif
