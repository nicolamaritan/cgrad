#ifndef TENSOR2D_MULT_LHS_TRANS_H
#define TENSOR2D_MULT_LHS_TRANS_H 

#include "cgrad/tensor/tensor.h"
#include "cgrad/error.h"

cgrad_error tensor2d_mult_lhs_trans_into(const struct tensor *const lhs_trans, const struct tensor *const rhs, struct tensor *const out);

#endif