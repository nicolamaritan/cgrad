#ifndef TENSOR_IM2ROW_H
#define TENSOR_IM2ROW_H

#include "cgrad/memory/allocators.h"
#include "cgrad/error.h"

cgrad_error tensor_im2row(struct tensor *t, const struct tensor *kernel, struct tensor **out, const bool track_grad, struct allocators *allocs);

#endif