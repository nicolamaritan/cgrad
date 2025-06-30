#ifndef INDEXES_BATCH_H
#define INDEXES_BATCH_H

#include "config.h"
#include <stddef.h>

/**
 * @struct indexes_batch
 * @brief Represents a linear datastructure for storing sampled indexes of samples of a dataset.
 *
 * It is a general abstraction for representing indexes of samples to be sampled from a dataset.
 */
typedef struct indexes_batch
{
   size_t *indexes;   /**< Pointer to the array of indexes. */
   size_t capacity;   /**< Maximum number of indexes that can be stored. */
   size_t size;       /**< Current number of indexes stored. */
} indexes_batch;

/**
 * @brief Allocates an indexes_batch container with initial size of 0.
 *
 * @param capacity Maximum number of indexes that can be stored in the container.
 * @return Pointer to the allocated indexes_batch, or NULL if allocation failed.
 */
indexes_batch *indexes_batch_alloc(const size_t capacity);

/**
 * @brief Frees the memory allocated for an indexes_batch container.
 *
 * @param ixs_batch Pointer to the indexes_batch to free.
 */
void indexes_batch_free(indexes_batch *ixs_batch);

#endif