#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "memory/allocators.h"

cgrad_error cross_entropy_loss(const struct tensor *const logits, const struct tensor *const targets, struct tensor **const loss, struct tensor_allocator *const tensor_alloc);
cgrad_error cross_entropy_loss_graph(struct tensor *const logits, struct tensor *const targets, struct tensor **const loss, struct allocators *const allocs);

#endif