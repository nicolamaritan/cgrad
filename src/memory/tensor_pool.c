#include "memory/tensor_pool.h"
#include "tensor/tensor.h"
#include "utils/error.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>

static inline void tensor_pool_init_chunks(struct tensor_pool *pool);

cgrad_error tensor_pool_init(struct tensor_pool *pool)
{
    if (!pool)
    {
        return MEMORY_POOL_NULL;
    }

    pool->tensor_chunk_head = calloc(MEMORY_TENSOR_POOL_N_CHUNKS, sizeof(struct tensor_pool_chunk) + sizeof(struct tensor));
    if (!pool->tensor_chunk_head)
    {
        return MEMORY_POOL_CHUNK_ALLOCATION_FAILED;
    }

    pool->data_chunk_head = calloc(MEMORY_TENSOR_POOL_N_CHUNKS, sizeof(struct tensor_pool_chunk) + MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE);
    if (!pool->data_chunk_head)
    {
        free(pool->tensor_chunk_head);
        return MEMORY_POOL_CHUNK_ALLOCATION_FAILED;
    }

    tensor_pool_init_chunks(pool);

    return NO_ERROR;
}

void *tensor_pool_tensor_alloc(struct tensor_pool *pool)
{
    void *return_ptr = (void *)pool->tensor_chunk_head + sizeof(struct tensor_pool_chunk);
    pool->tensor_chunk_head = pool->tensor_chunk_head->next;
    return return_ptr;
}

void *tensor_pool_data_alloc(struct tensor_pool *pool)
{
    void *return_ptr = (void *)pool->data_chunk_head + sizeof(struct tensor_pool_chunk);
    pool->data_chunk_head = pool->data_chunk_head->next;
    return return_ptr;
}

static inline void tensor_pool_init_chunks(struct tensor_pool *pool)
{
    struct tensor_pool_chunk *tensor_chunk_current = pool->tensor_chunk_head;
    struct tensor_pool_chunk *data_chunk_current = pool->data_chunk_head;

    for (size_t i = 0; i < MEMORY_TENSOR_POOL_N_CHUNKS - 1; i++)
    {
        tensor_chunk_current->next = (struct tensor_pool_chunk *)((char *)tensor_chunk_current + sizeof(struct tensor_pool_chunk) + sizeof(struct tensor));
        tensor_chunk_current = tensor_chunk_current->next;

        size_t lol = MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE;
        data_chunk_current->next = (struct tensor_pool_chunk *)((char *)data_chunk_current + sizeof(struct tensor_pool_chunk) + MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE);
        data_chunk_current = data_chunk_current->next;
    }
}