#ifndef COMPUTATIONAL_GRAPH_CPU_ALLOCATOR_H
#define COMPUTATIONAL_GRAPH_CPU_ALLOCATOR_H

#include "memory/computational_graph_allocator.h"
#include "memory/computational_graph_cpu_pool.h"

struct computational_graph_allocator make_computational_graph_cpu_allocator(struct computational_graph_cpu_pool *pool);

#endif