#ifndef LINEAR_H
#define LINEAR_H

#include "layers/linear_out.h"
#include "tensor/tensor.h"
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
    struct tensor_allocator *params_allocator;
    struct allocators *allocs;
};

struct linear *linear_alloc(const size_t in_dim, const size_t out_dim, const cgrad_dtype dtype, struct tensor_allocator *params_allocator, struct allocators *const allocs);
cgrad_error linear_forward(const struct tensor *const x, const struct linear *const layer, struct linear_out *const out);
cgrad_error linear_forward_graph(struct tensor *const x, struct linear *const layer, struct linear_out *const out);
cgrad_error linear_xavier_init(struct linear *layer);
void linear_free(struct linear *layer);

#endif