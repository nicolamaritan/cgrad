#ifndef TENSOR_CPU_ALLOCATOR_H
#define TENSOR_CPU_ALLOCATOR_H

#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_pool.h"

/**
 * @brief Initializes a tensor allocator with a CPU tensor pool.
 *
 * This function allocates and initializes a CPU tensor pool and sets up
 * the function pointers for allocation, deallocation, and cloning of tensors.
 *
 * @param[in,out] tensor_alloc Pointer to the tensor allocator to initialize.
 *
 * @return A cgrad_error code indicating the result of the operation:
 *         - NO_ERROR : Allocator successfully initialized
 *         - TENSOR_ALLOCATOR_NULL : tensor_alloc pointer is NULL
 *         - TENSOR_POOL_ALLOCATION_FAILED : Memory allocation for the CPU pool failed
 *         - Other error codes returned by tensor_cpu_pool_init()
 *
 * @note The tensor allocator must be properly cleaned up when no longer needed with 
 *       tensor_cpu_allocator_cleanup()
 * @note The CPU pool is managed internally by the allocator.
 */
cgrad_error tensor_cpu_allocator_init(struct tensor_allocator *const tensor_alloc);

/**
 * @brief Cleans up a tensor allocator and its CPU tensor pool.
 *
 * This function frees all memory associated with the tensor allocator's
 * CPU tensor pool and then frees the pool itself.
 *
 * @param[in,out] tensor_alloc Pointer to the tensor allocator to clean up.
 *
 * @note The tensor allocator and its pool must not be used after this function
 *       is called.
 * @note This function calls tensor_cpu_pool_cleanup() internally to free
 *       the pool's internal memory before freeing the pool structure itself.
 */
void tensor_cpu_allocator_cleanup(struct tensor_allocator *const tensor_alloc);

#endif