#ifndef LINEAR_H
#define LINEAR_H

#include "tensor/tensor.h"
#include "datastructures/tensor_list.h"
#include "memory/tensor/tensor_allocator.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/backpropagation/backpropagation.h"
#include "memory/allocators.h"
#include <stddef.h>

struct linear
{
    struct tensor *weights;
    struct tensor *biases;
    size_t in_dim;
    size_t out_dim;
    struct allocators *allocs;
};

cgrad_error linear_init(struct linear *const layer, const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct allocators *const allocs);
cgrad_error linear_forward(struct linear *const layer, struct tensor *const x, struct tensor **const out, struct tensor_list *const intermediates, const bool track_grad);
cgrad_error linear_xavier_init(struct linear *const layer);
void linear_cleanup(struct linear *const layer);

#endif