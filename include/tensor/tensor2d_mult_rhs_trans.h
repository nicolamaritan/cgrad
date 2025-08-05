#ifndef TENSOR2D_MULT_RHS_TRANS_H
#define TENSOR2D_MULT_RHS_TRANS_H 

#include "tensor/tensor.h"
#include "error.h"

cgrad_error tensor2d_mult_rhs_trans_into(const struct tensor *const lhs, const struct tensor *const rhs_trans, struct tensor *const out);

#endif