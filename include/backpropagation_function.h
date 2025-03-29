#ifndef BACKPROPAGATION_FUNCTION_H
#define BACKPROPAGATION_FUNCTION_H

#include "tensor.h"

typedef struct backpropagation_function_data backpropagation_function_data;

typedef void (*backpropagation_function_data_cleanup)(backpropagation_function_data* data);

typedef void (*backpropagation_function)(const backpropagation_function_data* const data, const tensor* const grad_wrt_out, tensor* grad_wrt_operand, size_t operand);

typedef struct backpropagation_function_data
{
    void* layer;
    void* inputs;
} backpropagation_function_data;

#endif