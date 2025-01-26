#ifndef MSE_H
#define MSE_H

#include "tensor.h"
#include "computational_graph.h"

typedef enum mse_loss_operand {
    MSE_PREDICTED = 0,
    MSE_TARGET = 1
} mse_loss_operand;

typedef struct
{
    tensor* predicted;
    tensor* target;
} mse_inputs;


void mse_loss(const tensor* const y_pred, const tensor* const y_target, tensor* const z);
void mse_loss_graph(tensor* const y_pred, tensor* const y_target, tensor* const out);
tensor* mse_loss_backpropagate(const backpropagation_function_data* const data, const tensor* const D, size_t operand);

#endif