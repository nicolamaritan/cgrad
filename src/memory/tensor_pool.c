#include "memory/tensor_pool.h"
#include "tensor/tensor.h"
#include "utils/error.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static void tensor_pool_init_chunks(struct tensor_pool *pool);

cgrad_error tensor_pool_init(struct tensor_pool *pool)
{
    if (!pool)
    {
        return MEMORY_POOL_NULL;
    }

    pool->tensor_memory= calloc(MEMORY_TENSOR_POOL_N_CHUNKS, sizeof(struct tensor_pool_tensor_chunk));
    if (!pool->tensor_memory)
    {
        return MEMORY_POOL_CHUNK_ALLOCATION_FAILED;
    }
    pool->tensor_chunk_head = (struct tensor_pool_tensor_chunk *)pool->tensor_memory;

    const size_t DATA_CHUNK_SIZE = sizeof(struct tensor_pool_data_chunk) + MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE;
    pool->data_memory = calloc(MEMORY_TENSOR_POOL_N_CHUNKS, DATA_CHUNK_SIZE);
    if (!pool->data_memory)
    {
        free(pool->tensor_memory);
        return MEMORY_POOL_CHUNK_ALLOCATION_FAILED;
    }
    pool->data_chunk_head = (struct tensor_pool_data_chunk *)pool->data_memory;

    tensor_pool_init_chunks(pool);
    return NO_ERROR;
}

void *tensor_pool_tensor_alloc(struct tensor_pool *pool)
{
    if (!pool || !pool->tensor_chunk_head)
    {
        return NULL;
    }

    struct tensor *return_ptr = &pool->tensor_chunk_head->t;
    pool->tensor_chunk_head = pool->tensor_chunk_head->next;
    return return_ptr;
}

void *tensor_pool_data_alloc(struct tensor_pool *pool)
{
    if (!pool || !pool->data_chunk_head)
    {
        return NULL;
    }

    // void *return_ptr = (void *)((char *)pool->data_chunk_head + offsetof(struct tensor_pool_data_chunk, data));
    void *return_ptr = (void *)pool->data_chunk_head->data;
    pool->data_chunk_head = pool->data_chunk_head->next;
    return return_ptr;
}

void *tensor_pool_data_zero_alloc(struct tensor_pool *pool)
{
    if (!pool || !pool->data_chunk_head)
    {
        return NULL;
    }

    void *return_ptr = (void *)pool->data_chunk_head->data;
    memset(return_ptr, 0, MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE);
    pool->data_chunk_head = pool->data_chunk_head->next;
    return return_ptr;
}

void tensor_pool_tensor_free(struct tensor_pool *pool, void *ptr)
{
    if (!pool || !ptr)
    {
        return;
    }

    struct tensor_pool_tensor_chunk *chunk = (struct tensor_pool_tensor_chunk *)((char *)ptr - offsetof(struct tensor_pool_tensor_chunk, t));
    chunk->next = pool->tensor_chunk_head;
    pool->tensor_chunk_head = chunk;
}

void tensor_pool_data_free(struct tensor_pool *pool, void *ptr)
{
    if (!pool || !ptr)
    {
        return;
    }

    struct tensor_pool_data_chunk *chunk = (struct tensor_pool_data_chunk *)((char *)ptr - offsetof(struct tensor_pool_data_chunk, data));
    assert(((size_t)((void*)chunk - pool->data_memory) % (sizeof(struct tensor_pool_data_chunk) + MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE)) == 0);
    chunk->next = pool->data_chunk_head;
    pool->data_chunk_head = chunk;
}

static void tensor_pool_init_chunks(struct tensor_pool *pool)
{
    struct tensor_pool_tensor_chunk *tensor_chunk_current = (struct tensor_pool_tensor_chunk *) pool->tensor_memory;
    struct tensor_pool_data_chunk *data_chunk_current = (struct tensor_pool_data_chunk *) pool->data_memory;

    for (size_t i = 0; i < MEMORY_TENSOR_POOL_N_CHUNKS - 1; i++)
    {
        tensor_chunk_current->next = (struct tensor_pool_tensor_chunk *)((char *)tensor_chunk_current + sizeof(struct tensor_pool_tensor_chunk));
        tensor_chunk_current = tensor_chunk_current->next;

        data_chunk_current->next = (struct tensor_pool_data_chunk *)((char *)data_chunk_current + sizeof(struct tensor_pool_data_chunk) + MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE);
        // printf("%td\n", (void*)data_chunk_current->next - (void*)data_chunk_current);
        data_chunk_current = data_chunk_current->next;
    }

    // Set the last chunk's next pointer to NULL
    tensor_chunk_current->next = NULL;
    data_chunk_current->next = NULL;
}