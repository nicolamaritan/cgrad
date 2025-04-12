#ifndef SGD_H
#define SGD_H

#include "autograd/backpropagation.h"
#include "model/model_params.h"

void sgd_step(double lr, model_params* params);

#endif