#ifndef COMPUTATIONAL_GRAPH_ALLOCATOR_H
#define COMPUTATIONAL_GRAPH_ALLOCATOR_H

#include "autograd/computational_graph.h"

typedef struct computational_graph_node *(*computational_graph_alloc_fn)(void *, struct tensor *const, struct tensor_allocator *);
typedef void (*computational_graph_free_fn)(void *, struct computational_graph_node *);

struct computational_graph_allocator
{
    computational_graph_alloc_fn alloc;
    computational_graph_free_fn free;
    void *pool;
};

static inline struct computational_graph_node *computational_graph_allocator_alloc(struct computational_graph_allocator *cg_allocator, struct tensor *const t, struct tensor_allocator *t_allocator);

static inline void computational_graph_allocator_free(struct computational_graph_allocator *allocator, struct computational_graph_node *ptr);

static inline struct computational_graph_node *computational_graph_allocator_alloc(struct computational_graph_allocator *cg_allocator, struct tensor *const t, struct tensor_allocator *t_allocator)
{
    return cg_allocator->alloc(cg_allocator->pool, t, t_allocator);
}

static inline void computational_graph_allocator_free(struct computational_graph_allocator *cg_allocator, struct computational_graph_node *ptr)
{
    cg_allocator->free(cg_allocator->pool, ptr);
}

#endif