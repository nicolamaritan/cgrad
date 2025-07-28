#ifndef BACKPROPAGATION_FUNCTION_H
#define BACKPROPAGATION_FUNCTION_H

#include "error.h"
#include "tensor/tensor.h"
#include "autograd/backpropagation/backpropagation_context.h"
#include <stdint.h>

/**
 * @typedef backpropagation_function
 * @brief Function pointer type for backpropagation functions.
 * 
 * @param ctx Pointer to the backpropagation context containing relevant tensors.
 * @param grad_wrt_out Gradient of the loss with respect to the output of the operation.
 * @param grad_wrt_operand Output tensor to store the computed gradient with respect to an operand.
 */
typedef cgrad_error (*backpropagation_function)(const struct backpropagation_context* const ctx, const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand);

static inline cgrad_error backpropagation_function_check_input(const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand);

static inline cgrad_error backpropagation_function_check_input(const struct tensor* const grad_wrt_out, struct tensor* grad_wrt_operand)
{
    if (!grad_wrt_operand || !grad_wrt_out)
    {
        return AUTOGRAD_BACKPROPAGATION_TENSOR_NULL;
    }
    if (grad_wrt_operand->dtype != grad_wrt_out->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    return NO_ERROR;
}


#endif