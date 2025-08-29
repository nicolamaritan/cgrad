#ifndef TENSOR_RESHAPE_H
#define TENSOR_RESHAPE_H

#include "cgrad/cgrad_env.h"
#include "cgrad/error.h"
#include <stddef.h>

cgrad_error tensor_reshape(struct tensor *const x, const size_t *shape, const size_t shape_size, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);
cgrad_error tensor_reshape_into(const struct tensor *const t, const size_t *shape, const size_t shape_size, struct tensor *const out);

#endif