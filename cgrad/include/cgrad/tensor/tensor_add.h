#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include "cgrad/cgrad_env.h"

cgrad_error tensor_add(struct tensor *const x, struct tensor *const y, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);

#endif