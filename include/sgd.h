#ifndef SGD_H
#define SGD_H

#include "backpropagation.h"
#include "model_params.h"

void sgd_step(double lr, model_params* params);

#endif