#ifndef CONV2D_H
#define CONV2D_H

#include "datastructures/tensor_list.h"
#include "tensor/tensor.h"
#include "memory/tensor/tensor_allocator.h"
#include "autograd/computational_graph/computational_graph.h"
#include "autograd/backpropagation/backpropagation.h"
#include "memory/allocators.h"
#include <stddef.h>

struct conv2d 
{
    struct tensor *weight;
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    struct tensor_allocator *params_allocator;
    struct allocators *allocs;
};

cgrad_error conv2d_init(struct conv2d *const layer, const size_t in_channels, const size_t out_channels, const size_t kernel_size, const cgrad_dtype dtype, struct tensor_allocator *const params_allocator, struct allocators *const allocs);
cgrad_error conv2d_forward(struct conv2d *const layer, struct tensor *const x, struct tensor **const out, struct tensor_list *const intermediates, const bool track_grad);
cgrad_error conv2d_xavier_init(struct conv2d *const layer);
void conv2d_cleanup(struct conv2d *const layer);

#endif