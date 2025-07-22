#include "autograd/backpropagation/backpropagation.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/backpropagation/backpropagation_queue.h"
#include "tensor/tensor_add_inplace.h"
#include "tensor/tensor_set.h"
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct backpropagation_targets
{
    struct computational_graph_node* targets[AUTOGRAD_MAX_TARGETS];
    size_t size;
};

static void build_gradients(struct computational_graph_node *loss_node, struct autograd_allocators *allocators, struct backpropagation_targets *targets);
static cgrad_error add_target(struct backpropagation_targets* const targets, struct computational_graph_node* const node);
static inline void set_gradient_wrt_itself(struct tensor* const t);

cgrad_error backward(struct tensor* t, struct autograd_allocators *allocators)
{
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (!allocators)
    {
        return AUTOGRAD_ALLOCATORS_NULL;
    }

    struct backpropagation_targets targets;
    targets.size = 0;

    set_gradient_wrt_itself(t);
    build_gradients(t->node, allocators, &targets);

    for (size_t i = 0; i < targets.size; i++)
    {
        struct computational_graph_node* node = targets.targets[i];
        node->t->node = NULL;
        computational_graph_allocator_free(allocators->cg_allocator, node);
    }

    return NO_ERROR;
}

static void build_gradients(struct computational_graph_node *loss_node, struct autograd_allocators *allocators, struct backpropagation_targets *targets)
{
    struct backpropagation_queue queue;
    backpropagation_queue_init(&queue);

    backpropagation_queue_push(&queue, loss_node);

    while (!backpropagation_queue_is_empty(&queue))
    {
        struct computational_graph_node *node = NULL;
        backpropagation_queue_pop(&queue, &node);
        add_target(targets, node);

        for (size_t i = 0; i < node->n_children; i++)
        {
            struct computational_graph_node *child_node = node->children[i];
            struct tensor *gradient = tensor_allocator_no_grad_alloc(allocators->t_allocator, child_node->t->shape, child_node->t->shape_size);
            struct backpropagation_context *ctx = &node->ctx;
            size_t operand = node->children_operands[i];

            node->function[operand](ctx, node->t->grad, gradient);
            tensor_add_inplace(child_node->t->grad, gradient);
            child_node->pushed_gradients_count++;

            tensor_allocator_free(allocators->t_allocator, gradient);

            if (child_node->pushed_gradients_count == child_node->n_parents)
            {
                backpropagation_queue_push(&queue, child_node);
            }
        }
    }
}

static cgrad_error add_target(struct backpropagation_targets* const targets, struct computational_graph_node* const node)
{
    size_t const size = targets->size;
    if (size >= AUTOGRAD_MAX_TARGETS)
    {
        return AUTOGRAD_MAX_TARGETS_EXCEEDED;
    }

    targets->targets[size] = node;
    targets->size++;

    return NO_ERROR;
}

static inline void set_gradient_wrt_itself(struct tensor* const t)
{
    tensor2d_set(t->grad, 0, 0, 1.0);
}