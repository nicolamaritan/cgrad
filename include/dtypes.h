#ifndef DTYPES_H
#define DTYPES_H

#include <inttypes.h>
#include <stddef.h>

typedef enum 
{
    DTYPE_FLOAT64,
    DTYPE_FLOAT32,
    DTYPE_INT32,
} cgrad_dtype;

static inline size_t dtype_sizeof(cgrad_dtype dtype);

static inline size_t dtype_sizeof(cgrad_dtype dtype)
{
    switch (dtype)
    {
        case DTYPE_FLOAT32:
            return sizeof(float);
        case DTYPE_FLOAT64:
            return sizeof(double);
        case DTYPE_INT32:
            return sizeof(int32_t);
        default:
            return 0;
    }
}

#endif