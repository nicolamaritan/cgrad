#ifndef SUM_LOSS_H
#define SUM_LOSS_H
 
#include "cgrad/cgrad_env.h"
 
cgrad_error sum_loss(struct tensor *const t, struct tensor **const z, const bool track_grad, struct cgrad_env *const env);
 
#endif
 