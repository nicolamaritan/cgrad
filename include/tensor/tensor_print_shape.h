#ifndef TENSOR_PRINT_SHAPE_H
#define TENSOR_PRINT_SHAPE_H

#include "tensor/tensor.h"
#include <stdio.h>

static inline cgrad_error tensor_print_shape(const struct tensor *const t);

static inline cgrad_error tensor_print_shape(const struct tensor *const t)
{
    printf("[");
    for (size_t i = 0; i < t->shape_size; i++)
    {
        printf("%ld", t->shape[i]);
        if (i + 1 < t->shape_size)
        {
            printf(", ");
        }
    }
    printf("]\n");

    return NO_ERROR;
}

#endif