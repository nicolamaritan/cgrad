#ifndef LINEAR_OUT_H
#define LINEAR_OUT_H

#include "memory/tensor/tensor_allocator.h"
#include "error.h"

struct linear_layer_out
{
    struct tensor *mult;
    struct tensor *result;
    struct tensor_allocator *tensor_alloc;
};

#define LINEAR_OUT_INIT {NULL, NULL, NULL}

static inline cgrad_error linear_layer_out_cleanup(struct linear_layer_out *const out);

static inline cgrad_error linear_layer_out_cleanup(struct linear_layer_out *const out)
{
    if (!out)
    {
        return LINEAR_OUT_NULL;
    }
    if (!out->tensor_alloc)
    {
        return TENSOR_ALLOCATOR_NULL;
    }
    tensor_allocator_free(out->tensor_alloc, out->mult);
    tensor_allocator_free(out->tensor_alloc, out->result);
    out->mult = NULL;
    out->result = NULL;
    out->tensor_alloc = NULL;

    return NO_ERROR;
}

#endif