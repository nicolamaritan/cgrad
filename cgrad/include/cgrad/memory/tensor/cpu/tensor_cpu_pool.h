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

/**
 * @brief Initialize a memory pool to allocate tensors
 * 
 * @param pool Pointer to tensor_cpu_pool struct used to manage the memory
 * 
 * @return Return a cgrad_error:
 *         - NO_ERROR no errors
 */
cgrad_error tensor_cpu_pool_init(struct tensor_cpu_pool *pool);

/**
 * @brief Allocate space to store tensor informations from the CPU pool
 * 
 * @param pool Pointer to a struct tensor_cpu_pool
 * 
 * @return void pointer to the allocated tensor
 */
void *tensor_cpu_pool_tensor_alloc(struct tensor_cpu_pool *pool);

/**
 * @brief Allocate space to store data from the CPU pool
 * 
 * @param pool Pointer to a struct tensor_cpu_pool
 * @param size Number of byte we want to allocate
 * 
 * @return void pointer to the allocated memory for data
 */
void *tensor_cpu_pool_data_alloc(struct tensor_cpu_pool *pool, const size_t size);

/**
 * @brief Allocate space to store data from the CPU pool, allocated memory is set to zero
 * 
 * @param pool Pointer to a struct tensor_cpu_pool
 * @param size Number of byte we want to allocate
 * 
 * @return void pointer to the allocated memory for data
 */
void *tensor_cpu_pool_data_zero_alloc(struct tensor_cpu_pool *pool, const size_t size);

/**
 * @brief Returns a tensor to the CPU tensor pool.
 *
 * This function does not free memory in the standard sense. Instead, it
 * reinserts the tensor into the pool's internal free list so that it can
 * be reused by future allocations.
 *
 * @param pool Pointer to the CPU tensor pool from which the tensor was allocated.
 * @param ptr  Pointer to the tensor to return to the pool.
 *
 * @note The pointer ptr must not be NULL. If pool or ptr is NULL, the function
 *       does nothing.
 */
void tensor_cpu_pool_tensor_free(struct tensor_cpu_pool *pool, void *ptr);

/**
 * @brief Returns a data block to the CPU data pool.
 *
 * This function does not free memory in the standard sense. Instead, it
 * reinserts the data block into the pool's internal free list so that it
 * can be reused by future allocations.
 *
 * @param pool Pointer to the CPU data pool from which the data block was allocated.
 * @param ptr  Pointer to the data block to return to the pool.
 *
 * @note The pointer ptr must not be NULL. If pool or ptr is NULL, the function
 *       does nothing.
 */
void tensor_cpu_pool_data_free(struct tensor_cpu_pool *pool, void *ptr);

/**
 * @brief Cleans up a CPU tensor pool.
 *
 * This function frees all memory associated with the CPU tensor pool,
 * including tensors and data blocks. After calling this function, the
 * pool is reset and cannot be used until reinitialized.
 *
 * @param[in,out] pool Pointer to the CPU tensor pool to clean up.
 *
 * @note This function frees all memory allocated for the pool. After calling
 *       it, the pool's internal pointers (tensor_memory, data_memory,
 *       tensor_chunk_head, data_chunk_head) are set to NULL.
 * @note Do not attempt to free individual tensors or data blocks from this
 *       pool after calling this function, as they are already freed.
 */
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
