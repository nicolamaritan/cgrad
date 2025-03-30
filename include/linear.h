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

linear_layer* linear_create(size_t in_dim, size_t out_dim);
void linear_forward_graph(tensor* const x, linear_layer* const layer, tensor *const mult, tensor* const out);
void linear_forward(const tensor* const x, const linear_layer* const layer, tensor *const mult, tensor* const out);
void linear_xavier_init(linear_layer* layer);
void linear_free(linear_layer* layer);

#endif