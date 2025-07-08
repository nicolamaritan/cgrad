#ifndef TENSOR_CPU_ALLOCATOR_H
#define TENSOR_CPU_ALLOCATOR_H

#include "memory/tensor_allocator.h"
#include "memory/tensor_cpu_pool.h"

struct tensor_allocator make_tensor_cpu_allocator(struct tensor_cpu_pool *pool);

#endif