#ifndef COMPUTATIONAL_GRAPH_LINK_H
#define COMPUTATIONAL_GRAPH_LINK_H

#include "cgrad/memory/allocators.h"
#include "cgrad/error.h"

/**
 * @brief Adds a link between two tensors in the computational graph.
 *
 * @param operand The operand tensor.
 * @param operand_id The ID of the operand.
 * @param result The result tensor.
 * @param backprop_function The backpropagation function to use.
 * @param TODO
 * @return NO_ERROR if successful, otherwise an appropriate error code.
 */
cgrad_error add_computational_graph_link(struct tensor* operand, size_t operand_id, struct tensor* result, backpropagation_function backprop_function, struct allocators *allocs);

#endif