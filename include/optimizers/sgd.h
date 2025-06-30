#ifndef SGD_H
#define SGD_H

#include "autograd/backpropagation.h"
#include "model/model_params.h"

struct sgd_state
{
    size_t size;
    struct tensor *prev_b_t[MODEL_MAX_PARAMS];
};

void sgd_step(double lr, double momentum, bool nesterov, struct sgd_state* state, struct model_params* params);
cgrad_error init_sgd_state(struct sgd_state *state, const struct model_params *const params);
void free_sgd_state_tensors(struct sgd_state *state);

#endif