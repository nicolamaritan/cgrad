#ifndef TENSOR_SOFTMAX_LAST_AXIS_H
#define TENSOR_SOFTMAX_LAST_AXIS_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include "cgrad/cgrad_env.h"

cgrad_error tensor_softmax_last_axis(struct tensor *const t, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);

#endif