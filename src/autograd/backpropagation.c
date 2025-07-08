#include "autograd/backpropagation.h"
#include "autograd/computational_graph.h"
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct backpropagation_targets
{
    struct computational_graph_node* targets[AUTOGRAD_MAX_TARGETS];
    size_t size;
};

static void identify_backpropagation_nodes(struct computational_graph_node* const node, struct backpropagation_targets* targets);
static struct tensor* build_gradient(struct computational_graph_node* const node);
static void build_gradients(struct backpropagation_targets* const targets);
static cgrad_error add_target(struct backpropagation_targets* const targets, struct computational_graph_node* const node);
static inline void set_gradient_wrt_itself(struct tensor* const t);

void backward(struct tensor* t, bool retain_graph)
{
    struct backpropagation_targets targets;
    targets.size = 0;

    identify_backpropagation_nodes(t->node, &targets);
    // printf("\n\n");

    set_gradient_wrt_itself(t);
    build_gradients(&targets);

    if (retain_graph)
    {
        return;
    }

    for (size_t i = 0; i < targets.size; i++)
    {
        struct computational_graph_node* node = targets.targets[i];
        node->t->node = NULL;
        free_computational_graph_node(node);
    }
}

static void identify_backpropagation_nodes(struct computational_graph_node* const node, struct backpropagation_targets* targets)
{
    node->is_involved_in_backprop = true;
    add_target(targets, node);
    // printf("%p\n", node->t);
    for (size_t i = 0; i < node->n_children; i++)
    {
        identify_backpropagation_nodes(node->children[i], targets);
    }
}

static struct tensor* build_gradient(struct computational_graph_node* const node)
{
    if (node->is_grad_computed)
    {
        return node->t->grad;
    }

    for (size_t i = 0; i < node->n_parents; i++)
    {
        if (!node->parents[i]->is_involved_in_backprop)
        {
            continue;
        }

        struct tensor* D = build_gradient(node->parents[i]);

        struct computational_graph_node *parent_node = node->parents[i];

        // Compute gradient and add to current grad
        struct tensor* parent_i_gradient = tensor_no_grad_alloc(node->t->shape, node->t->shape_size);

        // Retrieve context
        struct backpropagation_context *ctx = &parent_node->ctx;
        
        // Get which is the operand of the current node in the operation
        // that created the i-th parent. This info is stored in the current node
        size_t operand = node->parents_operands[i];
        parent_node->function[operand](ctx, D, parent_i_gradient);

        tensor_add_inplace(node->t->grad, parent_i_gradient);

        tensor_free(parent_i_gradient);
    }

    node->is_grad_computed = true;
    return node->t->grad;
}

static void build_gradients(struct backpropagation_targets* const targets)
{
    size_t size = targets->size;
    for (size_t i = 0; i < size; i++)
    {
        build_gradient(targets->targets[i]);
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
    tensor2d_set_unchecked(t->grad, 0, 0, 1);
}