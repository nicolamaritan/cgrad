#ifndef BACKPROPAGATION_CONTEXT_H
#define BACKPROPAGATION_CONTEXT_H

#include "utils/error.h"
#include "config.h"
#include <string.h>
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
 * This context stores intermediate tensors required by backpropagation functions.
 *
 * - `operands`: An array where each element points to a tensor allocated by the caller of the operation.
 *   The caller is responsible for deallocating these tensors.
 * 
 * - `owned`: An array where each element points to a tensor allocated by the forward function.
 *   These tensors are needed for backpropagation when they cannot be recomputed solely from the operands
 *   (e.g., results of random operations) or for performance reasons. All tensors pointed to by this array
 *   are deallocated by a dedicated cleanup function.
 *   The structure assumes that the first `n_owned` positions of the array 
 *   contain the owned tensors; otherwise, behavior is undefined.
 * 
 * - `n_owned`: The number of owned tensors currently stored in the context.
 */
struct backpropagation_context
{
    struct tensor *operands[AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE];
    struct tensor *owned[AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE];
    size_t n_owned;
};

/**
 * @brief Initialize a backpropagation context. Not invoking this causes
 * undefined behaviour.
 *
 * @param ctx Pointer to the backpropagation context.
 * @return cgrad_error Error code indicating success or failure.
 */
static inline cgrad_error context_init(struct backpropagation_context *const ctx);

/**
 * @brief Sets an operand in the backpropagation context at the specified ctx_id.
 * 
 * @param ctx Pointer to the backpropagation context.
 * @param t Pointer to the tensor to set.
 * @param ctx_id Index at which to store the tensor.
 * @return cgrad_error Error code indicating success or failure.
 */
static inline cgrad_error context_set_operand(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id);

/**
 * @brief Sets a tensor in the backpropagation context at the specified ctx_id.
 * The context takes ownership of the tensors set via this function.
 * 
 * @param ctx Pointer to the backpropagation context.
 * @param t Pointer to the tensor to set.
 * @param ctx_id Index at which to store the tensor.
 * @return cgrad_error Error code indicating success or failure.
 */
static inline cgrad_error context_set_owned(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id);

static inline cgrad_error context_init(struct backpropagation_context *const ctx)
{
    if (!ctx)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_NULL;
    }
    memset(ctx->operands, 0, sizeof(ctx->operands));
    memset(ctx->owned, 0, sizeof(ctx->owned));
    ctx->n_owned = 0;

    return NO_ERROR;
}

static inline void context_cleanup_owned(struct backpropagation_context *const ctx);

static inline cgrad_error context_set_operand(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id)
{
    if (ctx_id >= AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE)
    {
        return AUTOGRAD_INVALID_CONTEXT_ID;
    }

    ctx->operands[ctx_id] = t;
    return NO_ERROR;
}

static inline cgrad_error context_set_owned(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id)
{
    if (ctx_id >= AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE || ctx->owned[ctx_id] != NULL)
    {
        return AUTOGRAD_INVALID_CONTEXT_ID;
    }

    ctx->owned[ctx_id] = t;
    ctx->n_owned++;
    return NO_ERROR;
}

static inline void context_cleanup_owned(struct backpropagation_context *const ctx)
{
    for (size_t i = 0; i < ctx->n_owned; i++)
    {
        tensor_free(ctx->owned[i]);
    }
}

#endif
