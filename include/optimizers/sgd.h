#ifndef SGD_H
#define SGD_H

#include "autograd/backpropagation.h"
#include "model/model_params.h"

typedef struct sgd_state
{
    size_t size;
    tensor *prev_b_t[MODEL_MAX_PARAMS];
} sgd_state;

void sgd_step(double lr, double momentum, bool nesterov, sgd_state* state, model_params* params);
cgrad_error init_sgd_state(sgd_state *state, const model_params *const params);

#endif