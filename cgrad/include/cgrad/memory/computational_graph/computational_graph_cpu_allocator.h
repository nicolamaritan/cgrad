#ifndef COMPUTATIONAL_GRAPH_CPU_ALLOCATOR_H
#define COMPUTATIONAL_GRAPH_CPU_ALLOCATOR_H

#include "cgrad/memory/computational_graph/computational_graph_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_pool.h"

/**
 * @brief Initializes a computational graph allocator for CPU execution.
 *
 * This function sets up a `computational_graph_allocator` by allocating and 
 * initializing a CPU-based graph pool. If successful, it assigns the 
 * appropriate allocation and deallocation callbacks (`computational_graph_cpu_alloc` 
 * and `computational_graph_cpu_free`) and binds the pool to the allocator.
 *
 * @param graph_allocator Pointer to a `computational_graph_allocator` instance 
 *        to be initialized. Must not be NULL.
 *
 * @return 
 * - `NO_ERROR` if initialization succeeds.  
 * - `COMPUTATIONAL_GRAPH_ALLOCATOR_NULL` if `graph_allocator` is NULL.  
 * - `COMPUTATIONAL_GRAPH_POOL_ALLOCATION_FAILED` if the pool allocation fails.  
 * - An error code returned by `computational_graph_cpu_pool_init` if pool 
 *   initialization fails.
 */
cgrad_error computational_graph_cpu_allocator_init(struct computational_graph_allocator *const graph_allocator);

/**
 * @brief Cleans up a CPU-based computational graph allocator.
 *
 * This function releases resources associated with a
 * `computational_graph_allocator` previously initialized with
 * `computational_graph_cpu_allocator_init()`. It calls
 * `computational_graph_cpu_pool_cleanup()` on the underlying pool
 * and then frees the pool memory.
 *
 * @param graph_alloc Pointer to the `computational_graph_allocator`
 *        to clean up. If NULL, the function does nothing.
 */
void computational_graph_cpu_allocator_cleanup(struct computational_graph_allocator *const graph_alloc);

#endif