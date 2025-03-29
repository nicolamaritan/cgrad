#ifndef LINEAR
#define LINEAR

#include "tensor.h"
#include "computational_graph.h"
#include "backpropagation.h"
#include <stddef.h>

typedef struct {
    tensor* weights;
    tensor* biases;
    size_t in_dim;
    size_t out_dim;
} linear_layer;

typedef enum linear_layer_operand {
    PREDICTED = 0,
    WEIGHTS = 1,
    BIASES = 2,
} linear_layer_operand;

linear_layer* linear_create(size_t in_dim, size_t out_dim);
void linear_backpropagate(const backpropagation_function_data *const data, const tensor *const grad_wrt_out, tensor* grad_wrt_operand, size_t operand);
void linear_forward_graph(tensor* const x, linear_layer* const layer, tensor* const out);
void linear_forward(const tensor* const x, const linear_layer* const layer, tensor* const out);
void linear_xavier_init(linear_layer* layer);
void linear_free(linear_layer* layer);
void free_linear_backpropagation_function_data(backpropagation_function_data* data);


#endif