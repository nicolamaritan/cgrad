#ifndef TENSOR_CPU_ALLOCATOR_H
#define TENSOR_CPU_ALLOCATOR_H

#include "memory/tensor_allocator.h"
#include "memory/tensor_pool_alloc.h"

struct tensor_allocator make_tensor_cpu_allocator(struct tensor_pool *pool);

#endif