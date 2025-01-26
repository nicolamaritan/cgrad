#ifndef SGD_H
#define SGD_H

#include "computational_graph.h"

void sgd_step(double lr, target_computational_graph_nodes* targets);

#endif