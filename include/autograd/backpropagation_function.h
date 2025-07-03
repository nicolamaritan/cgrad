#ifndef BACKPROPAGATION_FUNCTION_H
#define BACKPROPAGATION_FUNCTION_H

#include "tensor/tensor.h"
#include "config.h"
#include <stdint.h>

/**
 * @typedef context_id
 * @brief Type used to index tensors in the backpropagation context.
 */
typedef uint8_t context_id;

/**
 * @struct backpropagation_context
 * @brief Holds pointers to tensors used during backpropagation.
 * 
 * The context stores intermediate tensors required by backpropagation functions.
 */
struct backpropagation_context
{
    struct tensor *tensors[AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE];
};

/**
 * @typedef backpropagation_function
 * @brief Function pointer type for backpropagation functions.
 * 
 * @param ctx Pointer to the backpropagation context containing relevant tensors.
 * @param grad_wrt_out Gradient of the loss with respect to the output of the operation.
 * @param grad_wrt_operand Output tensor to store the computed gradient with respect to an operand.
 */
typedef void (*backpropagation_function)(const struct backpropagation_context* const ctx, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand);

/**
 * @brief Sets a tensor in the backpropagation context at the specified id.
 * 
 * @param ctx Pointer to the backpropagation context.
 * @param t Pointer to the tensor to set.
 * @param id Index at which to store the tensor.
 * @return cgrad_error Error code indicating success or failure.
 */
static inline cgrad_error context_set_tensor(struct backpropagation_context *const ctx, struct tensor *t, const context_id id);

static inline cgrad_error context_set_tensor(struct backpropagation_context *const ctx, struct tensor *t, const context_id id)
{
    if (id >= AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE)
    {
        return AUTOGRAD_INVALID_CONTEXT_ID;
    }

    ctx->tensors[id] = t;
    return NO_ERROR;
}

#endif