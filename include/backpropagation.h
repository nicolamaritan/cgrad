#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "computational_graph.h"

typedef struct 
{
    tensor* targets[MAX_TARGETS];
    size_t size;
} backpropagation_targets;

void backpropagate(backpropagation_targets* const targets);
void zero_grad(tensor* const root);
int add_target(backpropagation_targets* const targets, tensor* const node);

#endif