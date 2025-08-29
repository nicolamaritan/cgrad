#ifndef TENSOR_TRANS_H
#define TENSOR_TRANS_H

#include "cgrad/cgrad_env.h"

cgrad_error tensor_trans(struct tensor *const t, const size_t axis_1, const size_t axis_2, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);

#endif 