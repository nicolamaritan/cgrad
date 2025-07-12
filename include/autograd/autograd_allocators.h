#ifndef AUTOGRAD_CONFIG_H
#define AUTOGRAD_CONFIG_H

#include "memory/tensor/tensor_allocator.h"
#include "memory/computational_graph/computational_graph_allocator.h"

struct autograd_allocators
{
    struct tensor_allocator *t_allocator;
    struct computational_graph_allocator *cg_allocator;
};

#endif