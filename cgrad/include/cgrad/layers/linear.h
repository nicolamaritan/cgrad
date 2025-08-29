#ifndef LINEAR_H
#define LINEAR_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/datastructures/tensor_list.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/cgrad_env.h"
#include <stddef.h>

struct linear
{
    struct tensor *weight;
    struct tensor *bias;
    size_t in_dim;
    size_t out_dim;
    struct cgrad_env *env;
};

cgrad_error linear_init(struct linear *const layer, const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct cgrad_env *const env);
cgrad_error linear_forward(struct linear *const layer, struct tensor *const x, struct tensor **const out, const bool track_grad);
cgrad_error linear_xavier_init(struct linear *const layer);
void linear_cleanup(struct linear *const layer);

#endif