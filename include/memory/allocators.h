#ifndef ALLOCATORS_H
#define ALLOCATORS_H

#include "memory/tensor/tensor_allocator.h"
#include "memory/computational_graph/computational_graph_allocator.h"

struct allocators
{
    struct tensor_allocator *tensor_alloc;
    struct computational_graph_allocator *graph_alloc;
};

#endif