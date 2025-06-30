#include "dataset/indexes_batch.h"
#include <stdlib.h>

/**
 * @brief Allocates an indexes_batch structure with the specified capacity.
 *
 * The batch is initially empty (size = 0).
 *
 * @param capacity Maximum number of indexes that can be stored.
 * @return Pointer to the allocated indexes_batch, or NULL if allocation failed.
 */
indexes_batch *indexes_batch_alloc(const size_t capacity)
{
    indexes_batch *ixs_batch = (indexes_batch*)malloc(sizeof(indexes_batch));
    if (!ixs_batch)
    {
        return NULL;
    }

    ixs_batch->indexes = (size_t*)calloc(capacity, sizeof(size_t));
    if (!ixs_batch->indexes)
    {
        free(ixs_batch);
        return NULL;
    }

    /***
     * Set capacity to the size of allocated memory,
     * while setting initial size to be zero, meaning the
     * container is empty.
     */
    ixs_batch->capacity = capacity;
    ixs_batch->size = 0;

    return ixs_batch;
}

/**
 * @brief Frees the memory allocated for an indexes_batch structure.
 *
 * @param ixs_batch Pointer to the indexes_batch to free.
 */
void indexes_batch_free(indexes_batch *ixs_batch)
{
    free(ixs_batch->indexes);
    free(ixs_batch);
}