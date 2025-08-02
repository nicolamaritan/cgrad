#ifndef LINEAR_H
#define LINEAR_H

#include "tensor/tensor.h"
#include "memory/tensor/tensor_allocator.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/backpropagation/backpropagation.h"
#include "memory/allocators.h"
#include <stddef.h>

struct linear_layer
{
    struct tensor *weights;
    struct tensor *biases;
    size_t in_dim;
    size_t out_dim;
    struct tensor_allocator *params_allocator;
    struct allocators *allocs;
};

struct linear_layer *linear_alloc(const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct tensor_allocator *params_allocator, struct allocators *const allocs);
cgrad_error linear_forward_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor **const out);
cgrad_error linear_forward(const struct tensor *const x, const struct linear_layer *const layer, struct tensor **const out);
cgrad_error linear_xavier_init(struct linear_layer *layer);
void linear_free(struct linear_layer *layer);

#endif