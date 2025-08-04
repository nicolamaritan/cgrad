#ifndef ALLOCATORS_H
#define ALLOCATORS_H

#include "memory/tensor/tensor_allocator.h"
#include "memory/computational_graph/computational_graph_allocator.h"

struct allocators
{
    struct tensor_allocator *tensor_alloc;
    struct computational_graph_allocator *graph_alloc;
};

static inline cgrad_error allocators_is_valid(struct allocators *allocs);

static inline cgrad_error allocators_is_valid(struct allocators *allocs)
{
    if (!allocs)
    {
        return ALLOCATORS_NULL;
    }
    if (!allocs->tensor_alloc)
    {
        return TENSOR_ALLOCATOR_NULL;
    }
    if (!allocs->graph_alloc)
    {
        return COMPUTATIONAL_GRAPH_ALLOCATOR_NULL;
    }

    return NO_ERROR;
}

#endif