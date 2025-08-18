#include "cgrad/tensor/tensor_helpers.h"
#include <stdio.h>

static void print_tensor_recursive_internal(const float *data, const size_t *shape, const size_t dimensions, const size_t offset);

void print_tensor(const struct tensor *const t)
{
    if (t == NULL || t->data == NULL)
    {
        printf("Invalid tensor\n");
        return;
    }
    size_t dimensions = 0;
    while (t->shape[dimensions] != 0)
    {
        dimensions++;
    }
    print_tensor_recursive_internal(t->data, t->shape, dimensions, 0);
    printf("\n");
}

static void print_tensor_recursive_internal(const float *data, const size_t *shape, const size_t dimensions, const size_t offset)
{
    if (dimensions == 1)
    {
        printf("[");
        for (size_t i = 0; i < shape[0]; i++)
        {
            printf("%.3lf", data[offset + i]);
            if (i < shape[0] - 1)
            {
                printf(", ");
            }
        }
        printf("]");
        return;
    }

    printf("[");
    for (size_t i = 0; i < shape[0]; i++)
    {
        print_tensor_recursive_internal(data, shape + 1, dimensions - 1, offset + i * shape[1]);
        if (i < shape[0] - 1)
        {
            printf(", ");
        }
    }
    printf("]");
}