#include "memory/computational_graph/computational_graph_cpu_allocator.h"
#include "memory/computational_graph/computational_graph_cpu_pool.h"
#include <string.h>

static struct computational_graph_node *computational_graph_cpu_alloc(void *pool, struct tensor *t);

static void computational_graph_cpu_free(void *pool, struct computational_graph_node *node);

struct computational_graph_allocator make_computational_graph_cpu_allocator(struct computational_graph_cpu_pool *pool)
{
    return (struct computational_graph_allocator){
        .alloc = computational_graph_cpu_alloc,
        .free = computational_graph_cpu_free,
        .pool = pool};
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
    // context_init(&node->ctx, t_allocator); // Pointer is not NULL at this point

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