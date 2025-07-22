#ifndef DTYPES_H
#define DTYPES_H

#include <inttypes.h>
#include <stddef.h>

typedef enum 
{
    DTYPE_FLOAT64,
    DTYPE_FLOAT32,
    DTYPE_INT32,
} dtype;

static inline size_t dtype_sizeof(dtype dt);

static inline size_t dtype_sizeof(dtype dt)
{
    switch (dt)
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