#include "dataset/index_permutation.h"
#include "utils/random.h"
#include <stdlib.h>

index_permutation *index_permutation_alloc(const size_t size)
{
    index_permutation *permutation = (index_permutation*)malloc(sizeof(index_permutation));
    size_t *index = (size_t*)malloc(size * sizeof(size_t));

    permutation->size = size;
    permutation->index = index;
    permutation->current = 0;
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

        // Swap the two elements
        size_t temp = index_permutation->index[random_index];
        index_permutation->index[random_index] = index_permutation->index[i];
        index_permutation->index[i] = temp;
    }

    return NO_ERROR;
}