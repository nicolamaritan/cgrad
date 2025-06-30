#ifndef INDEXES_BATCH_H
#define INDEXES_BATCH_H

#include "config.h"
#include <stddef.h>

typedef struct indexes_batch
{
   size_t capacity;
   size_t size;
   size_t *indexes;

} indexes_batch;

indexes_batch *indexes_batch_alloc(const size_t capacity);
void indexes_batch_free(indexes_batch *ixs_batch);

#endif