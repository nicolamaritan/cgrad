#ifndef LINEAR_H
#define LINEAR_H

#include "tensor/tensor.h"
#include "memory/tensor_allocator.h"
#include "autograd/computational_graph.h"
#include "autograd/backpropagation.h"
#include "autograd/autograd_allocators.h"
#include <stddef.h>

struct linear_layer
{
    struct tensor *weights;
    struct tensor *biases;
    size_t in_dim;
    size_t out_dim;
    struct tensor_allocator *params_allocator;
    struct autograd_allocators *ag_allocators;
};

struct linear_layer *linear_alloc(const size_t in_dim, const size_t out_dim, struct tensor_allocator *params_allocator, struct autograd_allocators *const ag_allocators);
cgrad_error linear_forward_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const out);
cgrad_error linear_forward(const struct tensor *const x, const struct linear_layer *const layer, struct tensor *const out);
void linear_xavier_init(struct linear_layer *layer);
void linear_free(struct linear_layer *layer);

#endif