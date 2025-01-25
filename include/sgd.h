#ifndef SGD_H
#define SGD_H

#include "computational_graph.h"
#include "grad_table.h"
#include "tensor.h"

void sgd_step(double lr, grad_table* table, target_computational_graph_nodes* targets);

#endif