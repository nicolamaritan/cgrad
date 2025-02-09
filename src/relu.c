#include "relu.h"
#include <stdlib.h>

tensor* relu_backpropagate(const backpropagation_function_data* const data, const tensor* const D, size_t operand)
{
    tensor* x = (tensor*)data->inputs;
    //tensor* out = tensor2d_alloc_like(x);
    tensor* out = tensor2d_no_grad_alloc(x->shape[0], x->shape[1]);
    
    // Avoid multiple indirections for performance
    double* x_data = x->data;
    double* out_data = out->data;
    size_t out_data_size = out->data_size;
    
    /*
        Gradient computation of dz/dX.
        dz/dX is the Hadamard Product of D = dz/drelu(X) and drelu(X)/dX,
        since element (i, j) of relu(X) depends only on element (i, j) of X.
    */
    
    for (size_t i = 0; i < out_data_size; i++)
    {
        // Element wise product
        out_data[i] = (x_data[i] > 0 ? 1 : 0) * D->data[i];
    }

    return out;
}

void relu_forward_graph(tensor* const x, tensor* const out)
{
    relu_forward(x, out);

    computational_graph_node* x_node = x->node ? x->node : computational_graph_node_tensor_alloc(x);
    computational_graph_node* out_node = computational_graph_node_tensor_alloc(out);

    add_parent(x_node, out_node, INPUT);
    add_child(out_node, x_node);

    backpropagation_function_data* data = malloc(sizeof(backpropagation_function_data));
    data->layer = NULL;
    data->inputs = (void*)x;
    out_node->data = data;

    backpropagation_function function = (backpropagation_function)&relu_backpropagate;
    out_node->function = function;
}

void relu_forward(const tensor* const x, tensor* const out)
{
    if (!tensor_same_shape(x, out))
    {
        return;
    }

    // Avoid multiple indirections for performance
    double* x_data = x->data;
    double* out_data = out->data;
    size_t out_data_size = out->data_size;

    for (size_t i = 0; i < out_data_size; i++)
    {
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }
}