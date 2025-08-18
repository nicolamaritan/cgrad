#include "cgrad/dataset/indexes_permutation.h"
#include "cgrad/utils/random.h"
#include <stdlib.h>

/**
 * @brief Swaps two indexes in the permutation.
 *
 * @param ixs_permutation Pointer to the permutation structure.
 * @param i Index of the first element.
 * @param j Index of the second element.
 */
static inline void indexes_permutation_swap(struct indexes_permutation *const ixs_permutation, const size_t i, const size_t j);

static inline void indexes_permutation_swap(struct indexes_permutation *const ixs_permutation, const size_t i, const size_t j)
{
    size_t temp = ixs_permutation->indexes[j];
    ixs_permutation->indexes[j] = ixs_permutation->indexes[i];
    ixs_permutation->indexes[i] = temp;
}

/**
 * @brief Allocates an indexes_permutation structure for the given size.
 *
 * @param size Number of indexes to permute.
 * @return Pointer to the allocated structure, or NULL if allocation failed.
 */
struct indexes_permutation *indexes_permutation_alloc(const size_t size)
{
    struct indexes_permutation *ixs_permutation = malloc(sizeof(struct indexes_permutation));
    if (!ixs_permutation)
    {
        return NULL;
    }

    size_t *index = (size_t*)malloc(size * sizeof(size_t));
    if (!index)
    {
        free(ixs_permutation);
        return NULL;
    }

    ixs_permutation->size = size;
    ixs_permutation->indexes = index;
    ixs_permutation->current = 0;

    return ixs_permutation;
}

/**
 * @brief Initializes the permutation with a random shuffle (Fisher-Yates).
 *
 * @param ixs_permutation Pointer to the permutation structure.
 * @return NO_ERROR on success, or an error code on failure.
 */
cgrad_error indexes_permutation_init(struct indexes_permutation* const ixs_permutation)
{
    // Fisher-Yates shuffles
    for (size_t i = 0; i < ixs_permutation->size; i++)
    {
        ixs_permutation->indexes[i] = i;
    }

    for (size_t i = 0; i < ixs_permutation->size; i++)
    {
        size_t random_index = sample_uniform_int(i, ixs_permutation->size - 1);
        indexes_permutation_swap(ixs_permutation, random_index, i);
    }

    return NO_ERROR;
}

/**
 * @brief Samples a batch of indexes from the permutation.
 *
 * Copies batch_size indexes from the current position into the batch container.
 *
 * @param ixs_permutation Pointer to the permutation structure.
 * @param ixs_batch Pointer to the batch container to fill.
 * @param batch_size Number of indexes to sample.
 * @return NO_ERROR on success, or an error code on failure.
 */
cgrad_error indexes_permutation_sample_index_batch(const struct indexes_permutation *const ixs_permutation, struct indexes_batch *const ixs_batch, const size_t batch_size)
{
    if (!ixs_permutation)
    {
        return INDEXES_PERMUTATION_NULL;
    }
    if (!ixs_batch)
    {
        return INDEXES_BATCH_NULL;
    }
    if (batch_size > ixs_batch->capacity || ixs_permutation->current + batch_size > ixs_permutation->size)
    {
        return INVALID_BATCH_SIZE;
    }

    // Update indexes in the container and update its size.
    memcpy(ixs_batch->indexes, ixs_permutation->indexes + ixs_permutation->current, batch_size * sizeof(size_t));
    ixs_batch->size = batch_size;

    return NO_ERROR;
}
