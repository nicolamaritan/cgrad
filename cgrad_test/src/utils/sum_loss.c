#include "cgrad_test/utils/sum_loss.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include "cgrad/tensor/tensor_helpers.h"
 
typedef enum sum_loss_operand
{
    SUM_TENSOR,
} sum_loss_operand;
 
static inline cgrad_error sum_loss_update_graph(struct tensor *const t, struct tensor **const z, struct cgrad_env *const env);
static cgrad_error sum_loss_dispatch(const struct tensor *const t, struct tensor *const z);
static cgrad_error sum_loss_f64(const struct tensor *const t, struct tensor *const z);
static cgrad_error sum_loss_f32(const struct tensor *const t, struct tensor *const z);
static cgrad_error sum_loss_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error sum_loss_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
static cgrad_error sum_loss_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);
 
cgrad_error sum_loss(struct tensor *const t, struct tensor **const z, const bool track_grad, struct cgrad_env *const env)
{
    if (!env)
    {
        return ALLOCATORS_NULL;
    }
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (!t->data)
    {
        return TENSOR_DATA_NULL;
    }
 
    const size_t shape[] = {1, 1};
    const size_t shape_size = 2;
    (*z) = tensor_allocator_alloc(&env->tensor_alloc, shape, shape_size, t->dtype);
 
    if (!(*z))
    {
        return TENSOR_ALLOCATION_FAILED;
    }
 
    cgrad_error err = sum_loss_dispatch(t, *z);
    if (err != NO_ERROR)
    {
        return err;
    }
 
    if (track_grad)
    {
        return sum_loss_update_graph(t, z, env);
    }
 
    return NO_ERROR;
}
 
static inline cgrad_error sum_loss_update_graph(struct tensor *const t, struct tensor **const z, struct cgrad_env *const env)
{
    return add_computational_graph_link(t, SUM_TENSOR, *z, &sum_loss_backpropagate, env);
}
 
static cgrad_error sum_loss_dispatch(const struct tensor *const t, struct tensor *const z)
{
    switch (t->dtype)
    {
    case DTYPE_FLOAT64:
        return sum_loss_f64(t, z);
    case DTYPE_FLOAT32:
        return sum_loss_f32(t, z);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}
 
static cgrad_error sum_loss_f64(const struct tensor *const t, struct tensor *const z)
{
    double *z_data = (double *)z->data;
    double *t_data = (double *)t->data;
 
    z_data[0] = 0;
    for (size_t i = 0; i < t->data_size; i++)
    {
        z_data[0] += t_data[i];
    }
 
    return NO_ERROR;
}
 
static cgrad_error sum_loss_f32(const struct tensor *const t, struct tensor *const z)
{
    float *z_data = (float *)z->data;
    float *t_data = (float *)t->data;
 
    z_data[0] = 0;
    for (size_t i = 0; i < t->data_size; i++)
    {
        z_data[0] += t_data[i];
    }
 
    return NO_ERROR;
}
 
static cgrad_error sum_loss_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    switch (grad_wrt_operand->dtype)
    {
    case DTYPE_FLOAT64:
        return sum_loss_backpropagate_f64(ctx, grad_wrt_out, grad_wrt_operand);
    case DTYPE_FLOAT32:
        return sum_loss_backpropagate_f32(ctx, grad_wrt_out, grad_wrt_operand);
    default:
        return AUTOGRAD_BACKPROPAGATION_INVALID_TENSOR_DTYPE;
    }
}
 
static cgrad_error sum_loss_backpropagate_f64(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    double *grad_wrt_operand_data = (double *)grad_wrt_operand->data;
    double grad_out = ((double *)grad_wrt_out->data)[0];
 
    // z = sum_i(t_i)  =>  dz/dt_i = 1 for every i, so every element of
    // the input receives exactly grad_out unchanged.
    for (size_t i = 0; i < grad_wrt_operand->data_size; i++)
    {
        grad_wrt_operand_data[i] = grad_out;
    }
 
    return NO_ERROR;
}
 
static cgrad_error sum_loss_backpropagate_f32(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    float *grad_wrt_operand_data = (float *)grad_wrt_operand->data;
    float grad_out = ((float *)grad_wrt_out->data)[0];
 
    for (size_t i = 0; i < grad_wrt_operand->data_size; i++)
    {
        grad_wrt_operand_data[i] = grad_out;
    }
 
    return NO_ERROR;
}
 