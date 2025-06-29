#include "dataset/index_batch.h"
#include <stdlib.h>

index_batch *index_batch_alloc(const size_t size)
{
    index_batch *ix_batch = (index_batch*)malloc(sizeof(index_batch));
    if (!ix_batch)
    {
        return NULL;
    }

    ix_batch->index = (size_t*)calloc(size, sizeof(size_t));
    if (!ix_batch->index)
    {
        free(ix_batch);
        return NULL;
    }

    ix_batch->size= size;

    return ix_batch;
}

void index_batch_free(index_batch *ix_batch)
{
    free(ix_batch->index);
    free(ix_batch);
}