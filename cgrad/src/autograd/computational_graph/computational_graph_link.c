#include "cgrad/autograd/computational_graph/computational_graph_link.h"

/**
 * @brief Adds a child node to a computational graph node.
 *
 * @param node The parent node.
 * @param child The child node to add.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
static cgrad_error add_child(struct computational_graph_node *const node, struct computational_graph_node *const child, const size_t operand);

/**
 * @brief Adds a parent node to a computational graph node.
 *
 * @param node The child node.
 * @param parent The parent node to add.
 * @param operand The operand associated with the parent.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
static cgrad_error add_parent(struct computational_graph_node *const node, struct computational_graph_node *const parent);

cgrad_error add_computational_graph_link(struct tensor *operand, size_t operand_id, struct tensor *result, backpropagation_function backprop_function, struct allocators *allocs)
{
    if (!operand || !result)
    {
        return TENSOR_NULL;
    }
    if (!operand->grad || !result->grad)
    {
        return TENSOR_GRAD_NULL;
    }
    if (!allocs)
    {
        return ALLOCATORS_NULL;
    }
    if (!backprop_function)
    {
        return AUTOGRAD_BACKPROPAGATION_FUNCTION_NULL;
    }

    cgrad_error err = NO_ERROR;

    if (!operand->node)
    {
        operand->node = computational_graph_allocator_alloc(allocs->graph_alloc, operand);
        if (!operand->node)
        {
            return AUTOGRAD_COMPUTATIONAL_GRAPH_NODE_ALLOCATION_ERROR;
        }
        if ((err = context_init(&operand->node->ctx, allocs->tensor_alloc)) != NO_ERROR)
        {
            return err;
        }
    }

    if (!result->node)
    {
        result->node = computational_graph_allocator_alloc(allocs->graph_alloc, result);
        if (!result->node)
        {
            computational_graph_allocator_free(allocs->graph_alloc, operand->node);
            return AUTOGRAD_COMPUTATIONAL_GRAPH_NODE_ALLOCATION_ERROR;
        }
        if ((err = context_init(&result->node->ctx, allocs->tensor_alloc)) != NO_ERROR)
        {
            return err;
        }
    }

    struct computational_graph_node *op_node = operand->node;
    struct computational_graph_node *res_node = result->node;

    // Setup connection
    if ((err = add_parent(op_node, res_node)) != NO_ERROR)
    {
        computational_graph_allocator_free(allocs->graph_alloc, op_node);
        computational_graph_allocator_free(allocs->graph_alloc, res_node);
        return err;
    }

    if ((err = add_child(res_node, op_node, operand_id)) != NO_ERROR)
    {
        computational_graph_allocator_free(allocs->graph_alloc, op_node);
        computational_graph_allocator_free(allocs->graph_alloc, res_node);
        return err;
    }

    // Setup backpropagation function
    result->node->function[operand_id] = backprop_function;

    // Setup operand in the tensor operands pointer
    context_set_operand(&res_node->ctx, operand, operand_id);

    return NO_ERROR;
}

static cgrad_error add_child(struct computational_graph_node *const node, struct computational_graph_node *const child, const size_t operand)
{
    size_t const n_children = node->n_children;
    if (n_children >= AUTOGRAD_MAX_CHILDREN)
    {
        return AUTOGRAD_MAX_CHILDREN_EXCEEDED;
    }

    node->children[n_children] = child;
    node->children_operands[n_children] = operand;
    node->n_children++;

    return NO_ERROR;
}

static cgrad_error add_parent(struct computational_graph_node *const node, struct computational_graph_node *const parent)
{
    size_t const n_parents = node->n_parents;
    if (n_parents >= AUTOGRAD_MAX_PARENTS)
    {
        return AUTOGRAD_MAX_PARENTS_EXCEEDED;
    }

    node->parents[n_parents] = parent;
    node->n_parents++;

    return NO_ERROR;
}