#include "memory/computational_graph_cpu_pool.h"
#include "utils/error.h"
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static void computational_graph_cpu_pool_init_chunks(struct computational_graph_cpu_pool *pool);

cgrad_error computational_graph_cpu_pool_init(struct computational_graph_cpu_pool *pool)
{
    if (!pool)
    {
        return MEMORY_POOL_NULL;
    }

    pool->memory = calloc(MEMORY_TENSOR_POOL_N_CHUNKS, sizeof(struct computational_graph_chunk));
    if (!pool->memory)
    {
        return MEMORY_POOL_CHUNK_ALLOCATION_FAILED;
    }
    pool->chunk_head = (struct computational_graph_chunk *)pool->memory;

    computational_graph_cpu_pool_init_chunks(pool);
    return NO_ERROR;
}

void *computational_graph_cpu_pool_alloc(struct computational_graph_cpu_pool *pool)
{
    if (!pool || !pool->chunk_head)
    {
        return NULL;
    }

    struct computational_graph_node *return_ptr = &pool->chunk_head->node;
    pool->chunk_head = pool->chunk_head->next;
    return return_ptr;
}

void computational_graph_cpu_pool_free(struct computational_graph_cpu_pool *pool, void *ptr)
{
    if (!pool || !ptr)
    {
        return;
    }

    struct computational_graph_chunk *chunk = (struct computational_graph_chunk *)((char *)ptr - offsetof(struct computational_graph_chunk, node));
    chunk->next = pool->chunk_head;
    pool->chunk_head = chunk;
}

static void computational_graph_cpu_pool_init_chunks(struct computational_graph_cpu_pool *pool)
{
    struct computational_graph_chunk *chunk_current = (struct computational_graph_chunk *) pool->memory;

    for (size_t i = 0; i < MEMORY_TENSOR_POOL_N_CHUNKS - 1; i++)
    {
        chunk_current->next = (struct computational_graph_chunk*)((char *)chunk_current + sizeof(struct computational_graph_chunk));
        chunk_current = chunk_current->next;
    }

    // Set the last chunk's next pointer to NULL
    chunk_current->next = NULL;
}