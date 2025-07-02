#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"

cgrad_error cross_entropy_loss(const struct tensor *const logits, const struct tensor *const targets, struct tensor *const loss);
cgrad_error cross_entropy_loss_graph(struct tensor *const logits, struct tensor *const targets, struct tensor *const loss);

#endif