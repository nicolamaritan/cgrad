#ifndef TENSOR_POOL_H
#define TENSOR_POOL_H

#include "cgrad/error.h"
#include "cgrad/tensor/tensor.h"
#include <stdalign.h>
#include <stdlib.h>

// Alignment for aligned SIMD
#define TENSOR_CPU_POOL_DATA_ALIGNMENT 32

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

    // alignas is needed to make sizeof(data_chunk) = 32
    alignas(TENSOR_CPU_POOL_DATA_ALIGNMENT) char data[];
};

struct tensor_cpu_pool
{
    struct tensor_chunk *tensor_chunk_head;
    struct data_chunk *data_chunk_head;
    void *tensor_memory;
    void *data_memory;
};

cgrad_error tensor_cpu_pool_init(struct tensor_cpu_pool *pool);
void *tensor_cpu_pool_tensor_alloc(struct tensor_cpu_pool *pool, const size_t size);
void *tensor_cpu_pool_data_alloc(struct tensor_cpu_pool *pool, const size_t size);
void *tensor_cpu_pool_data_zero_alloc(struct tensor_cpu_pool *pool, const size_t size);
void tensor_cpu_pool_tensor_free(struct tensor_cpu_pool *pool, void *ptr);
void tensor_cpu_pool_data_free(struct tensor_cpu_pool *pool, void *ptr);
static inline void tensor_cpu_pool_cleanup(struct tensor_cpu_pool *pool);

static inline void tensor_cpu_pool_cleanup(struct tensor_cpu_pool *pool)
{
    if (pool->tensor_memory)
    {
        free(pool->tensor_memory);
        pool->tensor_memory = NULL;
        pool->tensor_chunk_head = NULL;
    }

    if (pool->data_memory)
    {
        free(pool->data_memory);
        pool->data_memory = NULL;
        pool->data_chunk_head = NULL;
    }
}

#endif
