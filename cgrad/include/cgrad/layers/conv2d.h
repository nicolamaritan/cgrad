#ifndef CONV2D_H
#define CONV2D_H

#include "cgrad/datastructures/tensor_list.h"
#include "cgrad/tensor/tensor.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/cgrad_env.h"
#include <stddef.h>

struct conv2d 
{
    struct tensor *weight;
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    struct cgrad_env *env;
};

cgrad_error conv2d_init(struct conv2d *const layer, const size_t in_channels, const size_t out_channels, const size_t kernel_size, const cgrad_dtype dtype, struct cgrad_env *const env);
cgrad_error conv2d_forward(struct conv2d *const layer, struct tensor *const x, struct tensor **const out, const bool track_grad);
cgrad_error conv2d_xavier_init(struct conv2d *const layer);
void conv2d_cleanup(struct conv2d *const layer);

#endif