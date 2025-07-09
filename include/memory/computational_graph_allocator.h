#ifndef COMPUTATIONAL_GRAPH_ALLOCATOR_H
#define COMPUTATIONAL_GRAPH_ALLOCATOR_H

#include "autograd/computational_graph.h"

typedef struct computational_graph_node *(*computational_graph_alloc_fn)(void *, struct tensor *const);
typedef void (*computational_graph_free_fn)(void *, struct computational_graph_node *);

struct computational_graph_allocator
{
    computational_graph_alloc_fn alloc;
    computational_graph_free_fn free;
    void *pool;
};

static inline struct computational_graph_node *computational_graph_allocator_alloc(struct computational_graph_allocator *allocator, struct tensor *const t);
static inline void computational_graph_allocator_free(struct computational_graph_allocator *allocator, struct computational_graph_node *ptr);

static inline struct computational_graph_node *computational_graph_allocator_alloc(struct computational_graph_allocator *allocator, struct tensor *const t)
{
    return allocator->alloc(allocator->pool, t);
}

static inline void computational_graph_allocator_free(struct computational_graph_allocator *allocator, struct computational_graph_node *ptr)
{
    allocator->free(allocator->pool, ptr);
}

#endif