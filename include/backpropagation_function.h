#ifndef BACKPROPAGATION_FUNCTION_H
#define BACKPROPAGATION_FUNCTION_H

#include "tensor.h"

typedef struct backpropagation_function_data backpropagation_function_data;

typedef tensor* (*backpropagation_function)(const backpropagation_function_data* const data, const tensor* const G, size_t operand);

typedef struct backpropagation_function_data
{
    void* layer;
    void* inputs;
} backpropagation_function_data;

#endif