#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "computational_graph.h"
#include "grad_table.h"

void backpropagation(target_computational_graph_nodes* const targets, grad_table* const table);
tensor* build_grad(const computational_graph_node* const node, grad_table* const table);
void zero_grad(tensor* const root);
void zero_grad_node(computational_graph_node* const root);

#endif