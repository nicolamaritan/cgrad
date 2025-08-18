#ifndef TENSOR_CPU_ALLOCATOR_H
#define TENSOR_CPU_ALLOCATOR_H

#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_pool.h"

cgrad_error tensor_cpu_allocator_init(struct tensor_allocator *const tensor_alloc);
void tensor_cpu_allocator_cleanup(struct tensor_allocator *const tensor_alloc);

#endif