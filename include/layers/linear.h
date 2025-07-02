#ifndef LINEAR_H
#define LINEAR_H

#include "tensor/tensor.h"
#include "autograd/computational_graph.h"
#include "autograd/backpropagation.h"
#include <stddef.h>

typedef enum linear_layer_operand
{
    INPUT,
    WEIGHTS,
    BIAS,
} linear_layer_operand;

struct linear_layer
{
    struct tensor *weights;
    struct tensor *biases;
    size_t in_dim;
    size_t out_dim;
};

struct linear_layer *linear_alloc(size_t in_dim, size_t out_dim);
cgrad_error linear_forward_graph(struct tensor *const x, struct linear_layer *const layer, struct tensor *const out);
cgrad_error linear_forward(const struct tensor *const x, const struct linear_layer *const layer, struct tensor *const out);
void linear_xavier_init(struct linear_layer *layer);
void linear_free(struct linear_layer *layer);

void linear_backpropagate_input(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
void linear_backpropagate_weights(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
void linear_backpropagate_bias(const struct tensor **const operands, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

#endif