#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "computational_graph.h"

void backpropagate(target_computational_graph_nodes* const targets);
void zero_grad(tensor* const root);

#endif