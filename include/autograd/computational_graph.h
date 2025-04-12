#ifndef COMPUTATIONAL_GRAPH_H
#define COMPUTATIONAL_GRAPH_H

#include "autograd/backpropagation_function.h"
#include "utils/error.h"
#include "config.h"
#include <stdbool.h>

typedef struct computational_graph_node computational_graph_node;

/**
 * @struct computational_graph_node
 * @brief Represents a node in the computational graph used for automatic differentiation.
 *
 * This structure holds information about a node in the computational graph, including its tensor,
 * parents, children, and backpropagation functions. It is used to track the flow of data and
 * gradients during forward and backward passes in neural networks.
 */
typedef struct computational_graph_node
{
    tensor *t;                                   /**< Pointer to the tensor associated with this node. */
    size_t n_parents;                            /**< Number of parent nodes. */
    size_t n_children;                           /**< Number of child nodes. */
    computational_graph_node *parents[AUTOGRAD_MAX_PARENTS];  /**< Array of parent nodes. */
    size_t parents_operands[AUTOGRAD_MAX_PARENTS];            /**< Operands associated with each parent. */
    computational_graph_node *children[AUTOGRAD_MAX_CHILDREN];/**< Array of child nodes. */
    tensor *tensor_operands[AUTOGRAD_MAX_CHILDREN];             /**< Tensors pointed by the computational graph nodes of the children. This is used to avoid recomputation of the operands in build_gradient() in backpropagation.h even if redundant with children attribute.*/
    backpropagation_function function[AUTOGRAD_MAX_CHILDREN]; /**< Backpropagation functions for each child. */
    bool is_involved_in_backprop;                /**< Flag indicating if the node is involved in backpropagation. */
    bool is_grad_computed;                       /**< Flag indicating if the gradient has been computed. */
} computational_graph_node;

/**
 * @brief Allocates and initializes a new computational graph node.
 *
 * @return A pointer to the newly allocated computational graph node.
 */
computational_graph_node *computational_graph_node_alloc();

/**
 * @brief Allocates and initializes a new computational graph node with an associated tensor.
 *
 * @param t The tensor to associate with the new node.
 * @return A pointer to the newly allocated computational graph node.
 */
computational_graph_node *computational_graph_node_tensor_alloc(tensor *const t);

/**
 * @brief Frees the memory allocated for a computational graph node.
 *
 * @param node The node to free.
 */
void free_computational_graph_node(computational_graph_node *const node);

/**
 * @brief Adds a link between two tensors in the computational graph.
 *
 * @param operand The operand tensor.
 * @param operand_id The ID of the operand.
 * @param result The result tensor.
 * @param backprop_function The backpropagation function to use.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error add_computational_graph_link(tensor* operand, size_t operand_id, tensor* result, backpropagation_function backprop_function);

/**
 * @brief Adds a child node to a computational graph node.
 *
 * @param node The parent node.
 * @param child The child node to add.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error add_child(computational_graph_node *const node, computational_graph_node *const child);

/**
 * @brief Adds a parent node to a computational graph node.
 *
 * @param node The child node.
 * @param parent The parent node to add.
 * @param operand The operand associated with the parent.
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error add_parent(computational_graph_node *const node, computational_graph_node *const parent, const size_t operand);

/**
 * @brief Prints the details of a computational graph node.
 *
 * @param node The node to print.
 */
void print_computational_graph_node(const computational_graph_node *node);

#endif