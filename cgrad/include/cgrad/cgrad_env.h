#ifndef CGRAD_ENV_H 
#define CGRAD_ENV_H 

#include "cgrad/datastructures/tensor_list.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_allocator.h"

struct cgrad_env
{
    unsigned int seed;
    struct tensor_allocator tensor_alloc;
    struct tensor_list *tensor_alloc_intermediates;
    struct computational_graph_allocator graph_alloc;
};

/**
 * @brief Initializes the CGRAD environment.
 *
 * This function sets up the computational environment required for CGRAD,
 * including:
 * - Initializing the random seed.
 * - Initializing the CPU tensor allocator.
 * - Initializing the CPU computational graph allocator.
 * - Allocating a tensor list for intermediate tensors.
 *
 * If any step fails, the function performs cleanup of already-initialized
 * resources and returns an appropriate error code.
 *
 * @param env Pointer to the `cgrad_env` structure to initialize. Must not be NULL.
 * @param seed Random seed to use for initialization.
 * @param intermediates_capacity Initial capacity of the intermediate tensor list.
 *
 * @return 
 * - `NO_ERROR` if the environment was successfully initialized.
 * - Error codes from `tensor_cpu_allocator_init`, `computational_graph_cpu_allocator_init`,
 *   or allocation failures for intermediate tensors.
 *
 * @note If initialization fails, partially allocated resources are freed.
 */
cgrad_error cgrad_env_init(struct cgrad_env *env, const unsigned int seed, const size_t intermediates_capacity);

/**
 * @brief Cleans up the CGRAD environment.
 *
 * This function releases all resources allocated by `cgrad_env_init()`,
 * including:
 * - The CPU computational graph allocator.
 * - The CPU tensor allocator.
 * - The intermediate tensor list.
 *
 * @param env Pointer to the `cgrad_env` structure to clean up. Must not be NULL.
 *
 * @note This function does not free the `env` structure itself; it only
 *       cleans up its internal resources.
 */
void cgrad_env_cleanup(struct cgrad_env *env);

/**
 * @brief Frees all intermediate tensors in the CGRAD environment.
 *
 * This function iterates over the tensor list `tensor_alloc_intermediates`
 * in the provided environment and frees each tensor using `tensor_free()`.
 * After freeing, the list's size is reset to 0.
 *
 * @param env Pointer to the `cgrad_env` structure whose intermediate tensors
 *        should be freed. Must not be NULL.
 *
 * @return
 * - `NO_ERROR` if all intermediate tensors were successfully freed.
 * - `CGRAD_ENV_NULL` if `env` is NULL.
 *
 * @note This function does not free the intermediate tensor list itself,
 *       only the tensors contained in it.
 */
cgrad_error cgrad_env_free_intermediates(struct cgrad_env *env);

#endif