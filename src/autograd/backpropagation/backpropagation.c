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

static cgrad_error build_gradients(struct computational_graph_node *loss_node, struct allocators *allocs, struct backpropagation_targets *targets);
static cgrad_error add_target(struct backpropagation_targets* const targets, struct computational_graph_node* const node);
static inline cgrad_error set_gradient_wrt_itself(struct tensor* const t);

cgrad_error backward(struct tensor* t, struct allocators *allocs)
{
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (!allocs)
    {
        return AUTOGRAD_ALLOCATORS_NULL;
    }

    struct backpropagation_targets targets;
    targets.size = 0;

    cgrad_error err = NO_ERROR;
    if ((err = set_gradient_wrt_itself(t)) != NO_ERROR)
    {
        return err;
    }

    if ((err = build_gradients(t->node, allocs, &targets)) != NO_ERROR)
    {
        return err;
    }

    for (size_t i = 0; i < targets.size; i++)
    {
        struct computational_graph_node* node = targets.targets[i];
        node->t->node = NULL;
        computational_graph_allocator_free(allocs->graph_alloc, node);
    }

    return NO_ERROR;
}

static cgrad_error build_gradients(struct computational_graph_node *loss_node, struct allocators *allocs, struct backpropagation_targets *targets)
{
    cgrad_error err = NO_ERROR;

    struct backpropagation_queue queue;
    if ((err = backpropagation_queue_init(&queue)) != NO_ERROR)
    {
        return err;
    }

    if ((err = backpropagation_queue_push(&queue, loss_node)) != NO_ERROR)
    {
        return err;
    }

    while (!backpropagation_queue_is_empty(&queue))
    {
        struct computational_graph_node *node = NULL;
        backpropagation_queue_pop(&queue, &node);
        if ((err = add_target(targets, node)) != NO_ERROR)
        {
            return err;
        }

        for (size_t i = 0; i < node->n_children; i++)
        {
            struct computational_graph_node *child_node = node->children[i];
            struct tensor *gradient = tensor_allocator_no_grad_alloc(allocs->tensor_alloc, child_node->t->shape, child_node->t->shape_size, loss_node->t->dtype);
            if (!gradient)
            {
                return TENSOR_ALLOCATION_FAILED;
            }

            struct backpropagation_context *ctx = &node->ctx;
            size_t operand = node->children_operands[i];

            if ((err = backpropagation_function_check_input(node->t->grad, gradient)) != NO_ERROR)
            {
                return err;
            }

            if ((err = node->function[operand](ctx, node->t->grad, gradient)) != NO_ERROR)
            {
                return err;
            }

            if ((err = tensor_add_inplace(child_node->t->grad, gradient)) != NO_ERROR)
            {
                return err;
            }

            child_node->pushed_gradients_count++;

            tensor_allocator_free(allocs->tensor_alloc, gradient);

            if (child_node->pushed_gradients_count == child_node->n_parents)
            {
                if ((err = backpropagation_queue_push(&queue, child_node)) != NO_ERROR)
                {
                    return err;
                }
            }
        }
    }

    return NO_ERROR;
}

static cgrad_error add_target(struct backpropagation_targets* const targets, struct computational_graph_node* const node)
{
    if (!targets)
    {
        return AUTOGRAD_BACKPROPAGATION_TARGET_NULL;
    }
    if (!node)
    {
        return AUTOGRAD_BACKPROPAGATION_TARGET_COMP_GRAPH_NODE_NULL;
    }

    size_t const size = targets->size;
    if (size >= AUTOGRAD_MAX_TARGETS)
    {
        return AUTOGRAD_MAX_TARGETS_EXCEEDED;
    }

    targets->targets[size] = node;
    targets->size++;

    return NO_ERROR;
}

static inline cgrad_error set_gradient_wrt_itself(struct tensor* const t)
{
    switch (t->grad->dtype)
    {
        case DTYPE_FLOAT64:
            return tensor2d_set(t->grad, 0, 0, (double)1.0);
        case DTYPE_FLOAT32:
            return tensor2d_set(t->grad, 0, 0, (float)1.0);
        default:
            return AUTOGRAD_BACKPROPAGATION_INVALID_TENSOR_DTYPE;
    }
}