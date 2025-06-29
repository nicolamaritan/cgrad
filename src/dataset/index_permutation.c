#include "dataset/index_permutation.h"
#include "utils/random.h"
#include <stdlib.h>

static inline void index_permutation_swap(index_permutation *const index_permutation, const size_t i, const size_t j);

static inline void index_permutation_swap(index_permutation *const index_permutation, const size_t i, const size_t j)
{
    size_t temp = index_permutation->index[j];
    index_permutation->index[j] = index_permutation->index[i];
    index_permutation->index[i] = temp;
}

index_permutation *index_permutation_alloc(const size_t size)
{
    index_permutation *permutation = (index_permutation*)malloc(sizeof(index_permutation));
    if (!permutation)
    {
        return NULL;
    }

    size_t *index = (size_t*)malloc(size * sizeof(size_t));
    if (!index)
    {
        free(permutation);
        return NULL;
    }

    permutation->size = size;
    permutation->index = index;
    permutation->current = 0;

    return permutation;
}

cgrad_error index_permutation_init(index_permutation* const index_permutation)
{
    // Fisher-Yates shuffles
    for (size_t i = 0; i < index_permutation->size; i++)
    {
        index_permutation->index[i] = i;
    }

    for (size_t i = 0; i < index_permutation->size; i++)
    {
        size_t random_index = sample_uniform_int(i, index_permutation->size - 1);
        index_permutation_swap(index_permutation, random_index, i);
    }

    return NO_ERROR;
}

cgrad_error index_permutation_sample_index_batch(const index_permutation *const permutation, index_batch *const indeces, const size_t batch_size)
{
    if (!permutation)
    {
        return PERMUTATION_NULL;
    }
    if (!indeces)
    {
        return INDEX_BATCH_NULL;
    }
    if (batch_size > indeces->size)
    {
        return INVALID_BATCH_SIZE;
    }

    size_t missing_indexes = permutation->size - permutation->current;
    size_t effective_batch_size = missing_indexes < batch_size ? missing_indexes : batch_size;

    memcpy(indeces->index, permutation->index + permutation->current, effective_batch_size * sizeof(size_t));
    indeces->size = effective_batch_size;

    return NO_ERROR;
}
