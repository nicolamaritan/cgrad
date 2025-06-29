#ifndef INDEX_PERMUTATION
#define INDEX_PERMUTATION

#include "utils/error.h"
#include "index_batch.h"
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

typedef struct index_permutation
{
    size_t *index;
    size_t size;
    size_t current;
} index_permutation;

index_permutation *index_permutation_alloc(const size_t size);
cgrad_error index_permutation_init(index_permutation* const index_permutation);
cgrad_error index_permutation_sample_index_batch(const index_permutation *const index_permutation, index_batch *const index_batch, const size_t batch_size);
static inline void index_permutation_update(index_permutation *const permutation, const size_t batch_size);
static inline bool index_permutation_is_terminated(const index_permutation *const permutation);

static inline void index_permutation_update(index_permutation *const permutation, const size_t batch_size)
{
    permutation->current += batch_size;
    if (permutation->current >= permutation->size)
    {
        permutation->current = permutation->size;
    }
}

static inline bool index_permutation_is_terminated(const index_permutation *const permutation)
{
    return permutation->current == permutation->size;
}


#endif