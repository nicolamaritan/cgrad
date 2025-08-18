#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "cgrad/memory/allocators.h"

cgrad_error cross_entropy_loss(struct tensor *const logits, struct tensor *const targets, struct tensor **const loss, const bool track_grad, struct allocators *const allocs);

#endif