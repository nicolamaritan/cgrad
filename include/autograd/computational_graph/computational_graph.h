#ifndef COMPUTATIONAL_GRAPH_H
#define COMPUTATIONAL_GRAPH_H

#include "autograd/backpropagation/backpropagation_function.h"
#include "utils/error.h"
#include "config.h"
#include <stdbool.h>

struct computational_graph_node;

/**
 * @struct computational_graph_node
 * @brief Represents a node in the computational graph used for automatic differentiation.
 *
 * This structure holds information about a node in the computational graph, including its tensor,
 * parents, children, and backpropagation functions. It is used to track the flow of data and
 * gradients during forward and backward passes in neural networks.
 */
struct computational_graph_node
{
    struct tensor *t;                            /**< Pointer to the tensor associated with this node. */
    size_t n_parents;                            /**< Number of parent nodes. */
    size_t n_children;                           /**< Number of child nodes. */
    struct computational_graph_node *parents[AUTOGRAD_MAX_PARENTS];  /**< Array of parent nodes. */
    size_t parents_operands[AUTOGRAD_MAX_PARENTS];            /**< Operands associated with each parent. */
    struct computational_graph_node *children[AUTOGRAD_MAX_CHILDREN];/**< Array of child nodes. */
    backpropagation_function function[AUTOGRAD_MAX_CHILDREN]; /**< Backpropagation functions for each child. */
    struct backpropagation_context ctx;              /**< Context needed during backpropagation for computing gradients. */
    bool is_involved_in_backprop;                /**< Flag indicating if the node is involved in backpropagation. */
    bool is_grad_computed;                       /**< Flag indicating if the gradient has been computed. */
};

/**
 * @brief Prints the details of a computational graph node.
 *
 * @param node The node to print.
 */
void print_computational_graph_node(const struct computational_graph_node *node);

/**
 * @brief Sets a tensor in the context of a computational graph node at the specified context id.
 *
 * @param node Pointer to the computational graph node.
 * @param t Pointer to the tensor to set.
 * @param ctx_id Index at which to store the tensor in the node's context.
 * @return cgrad_error Error code indicating success or failure.
 */
static inline cgrad_error computational_graph_node_set_context_tensor(struct computational_graph_node *const node, struct tensor *t, const context_id ctx_id);

static inline cgrad_error computational_graph_node_set_context_tensor(struct computational_graph_node *const node, struct tensor *t, const context_id ctx_id)
{
    return context_set_operand(&node->ctx, t, ctx_id);
}

#endif
