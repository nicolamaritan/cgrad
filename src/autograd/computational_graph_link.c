#include "autograd/computational_graph_link.h"

/**
 * @brief Adds a child node to a computational graph node.
 *
 * @param node The parent node.
 * @param child The child node to add.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
static cgrad_error add_child(struct computational_graph_node *const node, struct computational_graph_node *const child);

/**
 * @brief Adds a parent node to a computational graph node.
 *
 * @param node The child node.
 * @param parent The parent node to add.
 * @param operand The operand associated with the parent.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
static cgrad_error add_parent(struct computational_graph_node *const node, struct computational_graph_node *const parent, const size_t operand);

cgrad_error add_computational_graph_link(struct tensor *operand, size_t operand_id, struct tensor *result, backpropagation_function backprop_function, struct autograd_allocators *allocators)
{
    if (!operand || !result)
    {
        return TENSOR_NULL;
    }
    if (!operand->grad || !result->grad)
    {
        return TENSOR_GRAD_NULL;
    }
    if (!allocators)
    {
        return AUTOGRAD_ALLOCATORS_NULL;
    }
    if (!backprop_function)
    {
        return AUTOGRAD_BACKPROPAGATION_FUNCTION_NULL;
    }

    struct computational_graph_node *operand_node = operand->node ? operand->node : computational_graph_allocator_alloc(allocators->cg_allocator, operand, allocators->t_allocator);
    if (!operand_node)
    {
        return AUTOGRAD_COMPUTATIONAL_GRAPH_NODE_ALLOCATION_ERROR;
    }

    struct computational_graph_node *result_node = result->node ? result->node : computational_graph_allocator_alloc(allocators->cg_allocator, result, allocators->t_allocator);
    if (!result_node)
    {
        computational_graph_allocator_free(allocators->cg_allocator, operand_node);
        return AUTOGRAD_COMPUTATIONAL_GRAPH_NODE_ALLOCATION_ERROR;
    }

    // Setup connection
    cgrad_error error = add_parent(operand_node, result_node, operand_id);
    if (error != NO_ERROR)
    {
        computational_graph_allocator_free(allocators->cg_allocator, operand_node);
        computational_graph_allocator_free(allocators->cg_allocator, result_node);
        return error;
    }

    error = add_child(result_node, operand_node);
    if (error != NO_ERROR)
    {
        computational_graph_allocator_free(allocators->cg_allocator, operand_node);
        computational_graph_allocator_free(allocators->cg_allocator, result_node);
        return error;
    }

    // Setup backpropagation function
    result_node->function[operand_id] = backprop_function;

    // Setup operand in the tensor operands pointer
    context_set_operand(&result_node->ctx, operand, operand_id);

    return NO_ERROR;
}

static cgrad_error add_child(struct computational_graph_node *const node, struct computational_graph_node *const child)
{
    size_t const n_children = node->n_children;
    if (n_children >= AUTOGRAD_MAX_CHILDREN)
    {
        return AUTOGRAD_MAX_CHILDREN_EXCEEDED;
    }

    node->children[n_children] = child;
    node->n_children++;

    return NO_ERROR;
}

static cgrad_error add_parent(struct computational_graph_node *const node, struct computational_graph_node *const parent, const size_t operand)
{
    size_t const n_parents = node->n_parents;
    if (n_parents >= AUTOGRAD_MAX_PARENTS)
    {
        return AUTOGRAD_MAX_PARENTS_EXCEEDED;
    }

    node->parents[n_parents] = parent;
    node->parents_operands[n_parents] = operand;
    node->n_parents++;

    return NO_ERROR;
}