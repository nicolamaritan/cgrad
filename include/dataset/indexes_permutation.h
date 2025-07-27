#ifndef INDEXES_PERMUTATION_H
#define INDEXES_PERMUTATION_H

#include "error.h"
#include "indexes_batch.h"
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

/**
 * @struct indexes_permutation
 * @brief Stores a permutation of indexes for sampling batches from a dataset.
 *
 * Used to shuffle and iterate through dataset indexes for batch sampling.
 */
struct indexes_permutation
{
    size_t *indexes;   /**< Array of permuted indexes. */
    size_t size;       /**< Total number of indexes. */
    size_t current;    /**< Current position in the permutation. */
};

/**
 * @brief Allocates an indexes_permutation structure.
 *
 * @param size Number of indexes to permute.
 * @return Pointer to the allocated indexes_permutation, or NULL if allocation failed.
 */
struct indexes_permutation *indexes_permutation_alloc(const size_t size);

/**
 * @brief Initializes the permutation with a random shuffle (Fisher-Yates).
 *
 * @param ixs_permutation Pointer to the indexes_permutation to initialize.
 * @return NO_ERROR on success, or an error code on failure.
 */
cgrad_error indexes_permutation_init(struct indexes_permutation* const ixs_permutation);

/**
 * @brief Samples a batch of indexes from the permutation.
 *
 * @param ixs_permutation Pointer to the permutation structure.
 * @param ixs_batch Pointer to the batch container to fill.
 * @param batch_size Number of indexes to sample.
 * @return NO_ERROR on success, or an error code on failure.
 */
cgrad_error indexes_permutation_sample_index_batch(const struct indexes_permutation *const ixs_permutation, struct indexes_batch *const ixs_batch, const size_t batch_size);

/**
 * @brief Advances the current position in the permutation by batch_size.
 *
 * @param ixs_permutation Pointer to the permutation structure.
 * @param batch_size Number of indexes to advance.
 */
static inline void index_permutation_update(struct indexes_permutation *const ixs_permutation, const size_t batch_size);

/**
 * @brief Checks if all indexes have been sampled.
 *
 * @param ixs_permutation Pointer to the permutation structure.
 * @return true if all indexes have been sampled, false otherwise.
 */
static inline bool index_permutation_is_terminated(const struct indexes_permutation *const ixs_permutation);

/**
 * @brief Gets the number of remaining indexes to sample.
 *
 * @param ixs_permutation Pointer to the permutation structure.
 * @return Number of remaining indexes.
 */
static inline size_t index_permutation_get_remaining(const struct indexes_permutation *const ixs_permutation);

static inline void index_permutation_update(struct indexes_permutation *const ixs_permutation, const size_t batch_size)
{
    ixs_permutation->current += batch_size;
    if (ixs_permutation->current >= ixs_permutation->size)
    {
        ixs_permutation->current = ixs_permutation->size;
    }
}

static inline bool index_permutation_is_terminated(const struct indexes_permutation *const ixs_permutation)
{
    return ixs_permutation->current == ixs_permutation->size;
}

static inline size_t index_permutation_get_remaining(const struct indexes_permutation *const ixs_permutation)
{
    return ixs_permutation->size - ixs_permutation->current;
}

#endif