#ifndef INDEX_BATCH_H
#define INDEX_BATCH_H

#include "config.h"
#include <stddef.h>

typedef struct index_batch
{
   size_t size;
   size_t *index;

} index_batch;

index_batch *index_batch_alloc(const size_t size);
void index_batch_free(index_batch *ix_batch);

#endif