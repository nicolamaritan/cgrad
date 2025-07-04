#ifndef TENSOR_POOL_H
#define TENSOR_POOL_H

#include "utils/error.h"

struct tensor_pool_chunk;
struct tensor_pool_chunk
{
    struct tensor_pool_chunk *next;
};

struct tensor_pool
{
    struct tensor_pool_chunk *tensor_chunk_head;
    struct tensor_pool_chunk *data_chunk_head; 
};

cgrad_error tensor_pool_init(struct tensor_pool *pool);
void* tensor_pool_tensor_alloc(struct tensor_pool *pool);
void* tensor_pool_data_alloc(struct tensor_pool *pool);

#endif
