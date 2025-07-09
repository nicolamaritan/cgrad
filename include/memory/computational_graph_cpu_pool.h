#ifndef COMPUTATIONAL_GRAPH_POOL_H
#define COMPUTATIONAL_GRAPH_POOL_H

#include "autograd/computational_graph.h"

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

cgrad_error computational_graph_cpu_pool_init(struct computational_graph_cpu_pool *pool);
void *computational_graph_cpu_pool_alloc(struct computational_graph_cpu_pool *pool);
void computational_graph_cpu_pool_free(struct computational_graph_cpu_pool *pool, void *ptr);

#endif
