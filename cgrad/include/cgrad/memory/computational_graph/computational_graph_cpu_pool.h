#ifndef COMPUTATIONAL_GRAPH_POOL_H
#define COMPUTATIONAL_GRAPH_POOL_H

#include "cgrad/autograd/computational_graph/computational_graph.h"
#include <stdlib.h>

struct computational_graph_chunk;
struct computational_graph_chunk
{
    struct computational_graph_chunk *next;
    struct computational_graph_node node;
};

struct computational_graph_cpu_pool
{
    struct computational_graph_chunk* chunk_head;
    void *memory;
};

/**
 * @brief Initializes a computational graph CPU memory pool.
 *
 * This function allocates memory for the computational graph pool and
 * initializes its internal free list of chunks.
 *
 * @param[in,out] pool Pointer to the computational graph CPU pool to initialize.
 *
 * @return A cgrad_error code indicating the result of the operation:
 *         - NO_ERROR : Pool successfully initialized
 *         - MEMORY_POOL_NULL : The pool pointer is NULL
 *         - MEMORY_POOL_CHUNK_ALLOCATION_FAILED : Memory allocation for chunks failed
 *
 * @note The pool must be properly cleaned up with computational_graph_cpu_pool_cleanup()
 *       when it is no longer needed.
 */
cgrad_error computational_graph_cpu_pool_init(struct computational_graph_cpu_pool *pool);

/**
 * @brief Allocates a node from the computational graph CPU pool.
 *
 * This function removes a node from the pool's internal free list
 * and returns a pointer to it. If the pool is empty or invalid,
 * NULL is returned.
 *
 * @param[in,out] pool Pointer to the computational graph CPU pool to allocate from.
 *
 * @return Pointer to a computational_graph_node from the pool,
 *         or NULL if the pool is empty or the pool pointer is NULL.
 *
 * @note The returned pointer should not be freed with free().
 *       It will be managed internally by the pool and should
 *       be returned to the pool using computational_graph_cpu_pool_free().
 */
void *computational_graph_cpu_pool_alloc(struct computational_graph_cpu_pool *pool);

/**
 * @brief Returns a node to the computational graph CPU pool.
 *
 * This function reinserts a node into the pool's internal free list so that
 * it can be reused by future allocations. It does not free memory with free().
 *
 * @param[in,out] pool Pointer to the computational graph CPU pool.
 * @param[in] ptr     Pointer to the node to return to the pool.
 *                    This should be a pointer previously obtained from
 *                    computational_graph_cpu_pool_alloc().
 *
 * @note The pointer ptr must not be NULL. If pool or ptr is NULL, the function
 *       does nothing.
 * @note The memory of the node is managed internally by the pool and should
 *       not be freed manually.
 */
void computational_graph_cpu_pool_free(struct computational_graph_cpu_pool *pool, void *ptr);

/**
 * @brief Cleans up a computational graph CPU pool.
 *
 * This function frees all memory associated with the computational graph pool
 * and resets the pool's internal pointers. After calling this function, the
 * pool cannot be used until reinitialized.
 *
 * @param[in,out] pool Pointer to the computational graph CPU pool to clean up.
 *
 * @note The pool's memory is freed and the internal pointers (memory, chunk_head)
 *       are set to NULL.
 * @note Do not attempt to free individual nodes from the pool after calling this
 *       function, as the memory has already been released.
 */
static inline void computational_graph_cpu_pool_cleanup(struct computational_graph_cpu_pool *pool);

static inline void computational_graph_cpu_pool_cleanup(struct computational_graph_cpu_pool *pool)
{
    if (!pool->memory)
    {
        return;
    }

    free(pool->memory);
    pool->memory = NULL;
    pool->chunk_head = NULL;
}

#endif
