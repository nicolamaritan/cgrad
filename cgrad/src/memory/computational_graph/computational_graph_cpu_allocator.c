#include "cgrad/memory/computational_graph/computational_graph_cpu_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_pool.h"
#include <string.h>

/**
 * @brief Allocates and initializes a computational graph node on the CPU.
 *
 * This function obtains a new `computational_graph_node` from the given CPU
 * graph pool, associates it with the provided tensor, and initializes its
 * bookkeeping fields (parents, children, gradients, etc.) to safe defaults.
 *
 * @param pool Pointer to a `computational_graph_cpu_pool` from which the node
 *        will be allocated. Must not be NULL.
 * @param t Pointer to the tensor that will be associated with the new node.
 *        Must not be NULL.
 *
 * @return Pointer to the newly allocated and initialized
 *         `computational_graph_node`, or NULL if allocation fails or if
 *         `t` is NULL.
 *
 * @note The function also sets `t->node` to point back to the allocated node.
 */
static struct computational_graph_node *computational_graph_cpu_alloc(void *pool, struct tensor *t);

/**
 * @brief Frees a computational graph node in a CPU-based graph pool.
 *
 * This function releases resources associated with a
 * `computational_graph_node`. It clears the back-reference from the
 * associated tensor (if any), cleans up the node's execution context, and
 * returns the node to the CPU pool for reuse.
 *
 * @param pool Pointer to a `computational_graph_cpu_pool` that owns the node.
 * @param node Pointer to the `computational_graph_node` to be freed.
 *
 * @note The function sets `node->t->node` to NULL if the tensor is still
 *       referencing the node.
 */
static void computational_graph_cpu_free(void *pool, struct computational_graph_node *node);

cgrad_error computational_graph_cpu_allocator_init(struct computational_graph_allocator *const graph_allocator)
{
    if (!graph_allocator)
    {
        return COMPUTATIONAL_GRAPH_ALLOCATOR_NULL;
    }

    struct computational_graph_cpu_pool *graph_pool = calloc(1, sizeof(struct computational_graph_cpu_pool));
    if (!graph_pool)
    {
        return COMPUTATIONAL_GRAPH_POOL_ALLOCATION_FAILED;
    }

    cgrad_error err = computational_graph_cpu_pool_init(graph_pool);
    if (err != NO_ERROR)
    {
        return err;
    }

    graph_allocator->alloc = computational_graph_cpu_alloc;
    graph_allocator->free = computational_graph_cpu_free;
    graph_allocator->pool = graph_pool;

    return NO_ERROR;
}

void computational_graph_cpu_allocator_cleanup(struct computational_graph_allocator *const graph_alloc)
{
    if (!graph_alloc)
    {
        return;
    }

    computational_graph_cpu_pool_cleanup(graph_alloc->pool);
    free(graph_alloc->pool);
}

static struct computational_graph_node *computational_graph_cpu_alloc(void *pool, struct tensor *t)
{ 
    struct computational_graph_cpu_pool *cpu_pool = (struct computational_graph_cpu_pool *)pool;
    struct computational_graph_node *node = computational_graph_cpu_pool_alloc(cpu_pool);
    if (!node || !t)
    {
        return NULL;
    }

    node->n_children = 0;
    node->n_parents = 0;
    node->t = t;
    t->node = node;
    node->is_involved_in_backprop = false;
    node->is_grad_computed = false;
    node->pushed_gradients_count = 0;

    // Initialize arrays to prevent undefined behavior
    memset(node->parents, 0, sizeof(node->parents));
    memset(node->children, 0, sizeof(node->children));
    // memset(node->parents_operands, 0, sizeof(node->parents_operands));
    memset(node->children_operands, 0, sizeof(node->children_operands));
    memset(node->function, 0, sizeof(node->function));
    // context_init(&node->ctx, tensor_alloc); // Pointer is not NULL at this point

    return node;
}

static void computational_graph_cpu_free(void *pool, struct computational_graph_node *node)
{
    struct computational_graph_cpu_pool *cpu_pool = (struct computational_graph_cpu_pool *)pool;

    if (node->t->node)
    {
        node->t->node = NULL;
    }

    context_cleanup_owned(&node->ctx);
    computational_graph_cpu_pool_free(cpu_pool, node);
}