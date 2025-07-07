#ifndef TENSOR_POOL_H
#define TENSOR_POOL_H

#include "utils/error.h"
#include "tensor/tensor.h"
#include <stdalign.h>

struct tensor_chunk;
struct tensor_chunk
{
    struct tensor_chunk *next;
    struct tensor t;
};

struct data_chunk;
struct data_chunk
{
    struct data_chunk *next;
    alignas(max_align_t) char data[];
};

struct tensor_pool
{
    struct tensor_chunk *tensor_chunk_head;
    struct data_chunk *data_chunk_head;
    void *tensor_memory;
    void *data_memory;
};

cgrad_error tensor_pool_init(struct tensor_pool *pool);
void *tensor_pool_tensor_alloc(struct tensor_pool *pool);
void *tensor_pool_data_alloc(struct tensor_pool *pool);
void *tensor_pool_data_zero_alloc(struct tensor_pool *pool);
void tensor_pool_tensor_free(struct tensor_pool *pool, void *ptr);
void tensor_pool_data_free(struct tensor_pool *pool, void *ptr);

#endif
