#ifndef TENSOR_RESHAPE_H
#define TENSOR_RESHAPE_H

#include "memory/allocators.h"
#include "error.h"
#include <stddef.h>

cgrad_error tensor_reshape(struct tensor *const x, const size_t *shape, const size_t shape_size, struct tensor **const out, const bool track_grad, struct allocators *const allocs);
cgrad_error tensor_reshape_into(const struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor *const out);

#endif