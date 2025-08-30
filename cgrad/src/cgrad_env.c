#include "cgrad/cgrad_env.h"
#include "cgrad/utils/random.h"
#include "cgrad/tensor/tensor_alloc.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_allocator.h"

cgrad_error cgrad_env_init(struct cgrad_env *env, const unsigned int seed, const size_t intermediates_capacity)
{
    init_random_seed(seed);

    cgrad_error err = NO_ERROR;
    err = tensor_cpu_allocator_init(&env->tensor_alloc);
    if (err != NO_ERROR)
    {
        goto tensor_allocator_init_fail;
    }

    err = computational_graph_cpu_allocator_init(&env->graph_alloc);
    if (err != NO_ERROR)
    {
        goto computational_graph_allocator_init_failed;
    }

    env->tensor_alloc_intermediates = tensor_list_alloc(intermediates_capacity);
    if (!env->tensor_alloc_intermediates)
    {
        goto tensor_intermediates_allocation_failed;
    }

    return NO_ERROR;

tensor_intermediates_allocation_failed:
    computational_graph_cpu_allocator_cleanup(&env->graph_alloc);
computational_graph_allocator_init_failed:
    tensor_cpu_allocator_cleanup(&env->tensor_alloc);
tensor_allocator_init_fail:
    return err;
}

void cgrad_env_cleanup(struct cgrad_env *env)
{
    computational_graph_cpu_allocator_cleanup(&env->graph_alloc);
    tensor_cpu_allocator_cleanup(&env->tensor_alloc);
    tensor_list_free(env->tensor_alloc_intermediates);
}

cgrad_error cgrad_env_free_intermediates(struct cgrad_env *env)
{
    if (!env)
    {
        return CGRAD_ENV_NULL;
    }

    for (size_t i = 0; i < env->tensor_alloc_intermediates->size; i++)
    {
        tensor_free(env, env->tensor_alloc_intermediates->data[i]);
    }

    env->tensor_alloc_intermediates->size = 0;

    return NO_ERROR;
}