#ifndef COMPUTATIONAL_GRAPH_ALLOCATOR_H
#define COMPUTATIONAL_GRAPH_ALLOCATOR_H

#include "cgrad/autograd/computational_graph/computational_graph.h"

typedef struct computational_graph_node *(*computational_graph_alloc_fn)(void *, struct tensor *const);
typedef void (*computational_graph_free_fn)(void *, struct computational_graph_node *);

struct computational_graph_allocator
{
    computational_graph_alloc_fn alloc;
    computational_graph_free_fn free;
    void *pool;
};

static inline struct computational_graph_node *computational_graph_allocator_alloc(struct computational_graph_allocator *graph_alloc, struct tensor *const t);

static inline void computational_graph_allocator_free(struct computational_graph_allocator *allocator, struct computational_graph_node *ptr);

/**
 * @brief Allocates a computational graph node using the given allocator.
 *
 * This function calls the allocator's `alloc` callback, passing in its
 * internal pool and the specified tensor. The newly created node will be
 * associated with the tensor.
 *
 * @param graph_alloc Pointer to a `computational_graph_allocator` that manages
 *        allocation. Must not be NULL.
 * @param t Pointer to the tensor to associate with the new node. Must not be NULL.
 *
 * @return Pointer to the allocated `computational_graph_node`, or NULL if
 *         allocation fails.
 */
static inline struct computational_graph_node *computational_graph_allocator_alloc(struct computational_graph_allocator *graph_alloc, struct tensor *const t)
{
    return graph_alloc->alloc(graph_alloc->pool, t);
}

/**
 * @brief Frees a computational graph node using the given allocator.
 *
 * This function calls the allocator's `free` callback, passing in its
 * internal pool and the node to be released.
 *
 * @param graph_alloc Pointer to a `computational_graph_allocator` that manages
 *        memory. Must not be NULL.
 * @param ptr Pointer to the `computational_graph_node` to free. Must not be NULL.
 */
static inline void computational_graph_allocator_free(struct computational_graph_allocator *graph_alloc, struct computational_graph_node *ptr)
{
    graph_alloc->free(graph_alloc->pool, ptr);
}

#endif