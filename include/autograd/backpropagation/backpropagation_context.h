#ifndef BACKPROPAGATION_CONTEXT_H
#define BACKPROPAGATION_CONTEXT_H

#include "memory/tensor/tensor_allocator.h"
#include "error.h"
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
    size_t operands_size_t[AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE];
    struct tensor *owned[AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE];
    size_t n_owned;
    struct tensor_allocator *owned_allocator;
};

// --- Function declarations ---

/**
 * @brief Initializes a backpropagation context for use in autograd operations.
 *
 * This function must be called before using the context. It sets all operand and owned
 * tensor pointers to NULL, resets the owned tensor count, and assigns the provided
 * tensor allocator for owned tensors.
 *
 * @param ctx Pointer to the backpropagation context to initialize.
 * @param autograd_tensor_allocator Allocator to use for owned tensors.
 * @return cgrad_error Error code indicating success or failure.
 *         - NO_ERROR on success.
 *         - AUTOGRAD_BACKPROPAGATION_CONTEXT_NULL if ctx is NULL.
 *         - TENSOR_ALLOCATOR_NULL if autograd_tensor_allocator is NULL.
 */
static inline cgrad_error context_init(struct backpropagation_context *const ctx, struct tensor_allocator *autograd_tensor_allocator);

/**
 * @brief Stores a tensor pointer as an operand in the context at the given index.
 *
 * The context does not take ownership of the tensor; the caller is responsible for
 * deallocating it. Overwrites any existing pointer at the specified index.
 *
 * @param ctx Pointer to the backpropagation context.
 * @param t Pointer to the tensor to store.
 * @param ctx_id Index at which to store the tensor (must be less than AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE).
 * @return cgrad_error Error code indicating success or failure.
 *         - NO_ERROR on success.
 *         - AUTOGRAD_BACKPROPAGATION_CONTEXT_NULL if ctx is NULL.
 *         - AUTOGRAD_INVALID_CONTEXT_ID if ctx_id is out of bounds.
 *         - TENSOR_NULL if t is NULL.
 */
static inline cgrad_error context_set_operand(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id);

/**
 * TODO add docs
 */
static inline cgrad_error context_set_operand_size_t(struct backpropagation_context *const ctx, const size_t op, const context_id ctx_id);

/**
 * @brief Stores a tensor pointer as an owned tensor in the context at the given index.
 *
 * The context takes ownership of the tensor and will deallocate it during cleanup.
 * Fails if the index is already occupied.
 *
 * @param ctx Pointer to the backpropagation context.
 * @param t Pointer to the tensor to store.
 * @param ctx_id Index at which to store the tensor (must be less than AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE).
 * @return cgrad_error Error code indicating success or failure.
 *         - NO_ERROR on success.
 *         - AUTOGRAD_INVALID_CONTEXT_ID if ctx_id is out of bounds.
 *         - AUTOGRAD_CONTEXT_ID_ALREADY_TAKEN if the slot is already occupied.
 */
static inline cgrad_error context_set_owned(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id);

/**
 * @brief Frees all owned tensors in the context using the assigned allocator.
 *
 * Iterates over the first n_owned entries in the owned array and frees each tensor.
 * Does nothing if ctx is NULL.
 *
 * @param ctx Pointer to the backpropagation context.
 */
static inline void context_cleanup_owned(struct backpropagation_context *const ctx);


// --- Function definitions ---

static inline cgrad_error context_init(struct backpropagation_context *const ctx, struct tensor_allocator *autograd_tensor_allocator)
{
    if (!ctx)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_NULL;
    }
    if (!autograd_tensor_allocator)
    {
        return TENSOR_ALLOCATOR_NULL;
    }

    memset(ctx->operands, 0, sizeof(ctx->operands));
    memset(ctx->owned, 0, sizeof(ctx->owned));
    memset(ctx->operands_size_t, 0, sizeof(ctx->operands_size_t));
    ctx->n_owned = 0;
    ctx->owned_allocator = autograd_tensor_allocator;

    return NO_ERROR;
}

static inline cgrad_error context_set_operand(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id)
{
    if (!ctx)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_NULL;
    }
    if (ctx_id >= AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE)
    {
        return AUTOGRAD_INVALID_CONTEXT_ID;
    }
    if (!t)
    {
        return TENSOR_NULL;
    }

    ctx->operands[ctx_id] = t;
    return NO_ERROR;
}

static inline cgrad_error context_set_operand_size_t(struct backpropagation_context *const ctx, const size_t op, const context_id ctx_id)
{
    if (!ctx)
    {
        return AUTOGRAD_BACKPROPAGATION_CONTEXT_NULL;
    }
    if (ctx_id >= AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE)
    {
        return AUTOGRAD_INVALID_CONTEXT_ID;
    }

    ctx->operands_size_t[ctx_id] = op;
    return NO_ERROR;
}

static inline cgrad_error context_set_owned(struct backpropagation_context *const ctx, struct tensor *t, const context_id ctx_id)
{
    if (ctx_id >= AUTOGRAD_MAX_BACKPROPAGATION_FUNCTION_CONTEXT_SIZE)
    {
        return AUTOGRAD_INVALID_CONTEXT_ID;
    }
    if (ctx->owned[ctx_id])
    {
        return AUTOGRAD_CONTEXT_ID_ALREADY_TAKEN;
    }

    ctx->owned[ctx_id] = t;
    ctx->n_owned++;
    return NO_ERROR;
}

static inline void context_cleanup_owned(struct backpropagation_context *const ctx)
{
    if (!ctx)
    {
        return;
    }

    for (size_t i = 0; i < ctx->n_owned; i++)
    {
        tensor_allocator_free(ctx->owned_allocator, ctx->owned[i]);
    }
}

#endif