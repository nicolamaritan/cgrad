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

cgrad_error cgrad_env_init(struct cgrad_env *env, const unsigned int seed, const size_t intermediates_capacity);
void cgrad_env_cleanup(struct cgrad_env *env);
cgrad_error cgrad_env_free_intermediates(struct cgrad_env *env);

#endif