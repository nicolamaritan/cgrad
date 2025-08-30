#include "cgrad/optimizers/sgd.h"
#include "cgrad/tensor/tensor_add_inplace.h"
#include "cgrad/tensor/tensor_scalar_mult_tensor_add.h"
#include "cgrad/tensor/tensor_axpy.h"

static cgrad_error add_prev_b_t(struct sgd_optimizer *const opt, struct tensor *const prev_grad);

cgrad_error sgd_optimizer_init(struct sgd_optimizer *opt, struct model_params *const params, const double lr, const double momentum, const bool nesterov, struct cgrad_env *env)
{
    if (!opt)
    {
        return OPTIMIZER_NULL;
    }
    if (!params)
    {
        return MODEL_PARAMS_NULL;
    }
    if (!env)
    {
        return CGRAD_ENV_NULL;
    }

    opt->lr = lr;
    opt->momemtum = momentum;
    opt->nesterov = nesterov;
    opt->params = params;
    opt->tensor_alloc = &env->tensor_alloc;
    opt->size = 0;
    for (size_t i = 0; i < params->size; i++)
    {
        struct tensor* param = params->params[i];
        struct tensor* param_prev_grad = tensor_allocator_no_grad_zero_alloc(opt->tensor_alloc, param->shape, param->shape_size, param->dtype);

        cgrad_error err = add_prev_b_t(opt, param_prev_grad);
        if (err != NO_ERROR)
        {
            return err;
        }
    }

    return NO_ERROR;
}

cgrad_error sgd_optimizer_step(struct sgd_optimizer *opt)
{
    if (!opt)
    {
        return OPTIMIZER_NULL;
    }

    double lr = opt->lr;
    double momentum = opt->momemtum;
    bool nesterov = opt->nesterov;

    for (size_t i = 0; i < opt->params->size; i++)
    {
        struct tensor* param = opt->params->params[i];
        struct tensor_allocator *tensor_alloc = opt->tensor_alloc;
        
        struct tensor* prev_b_t = opt->prev_b_t[i];
        struct tensor* b_t = tensor_allocator_no_grad_alloc(tensor_alloc, prev_b_t->shape, prev_b_t->shape_size, param->dtype);

        if (momentum != 0)
        {
            if (nesterov)
            {
                // b_t <- momentum * b_t-1 + g_t
                struct tensor* g_t = tensor_allocator_clone(tensor_alloc, param->grad);
                tensor_scalar_mult_tensor_add(prev_b_t, g_t, momentum, b_t);

                // g_t <- g_t + momentum * b_t
                tensor_axpy(b_t, g_t, momentum);

                // SGD update using g_t, i.e.:
                // param <- param - lr * g_t
                tensor_axpy(g_t, param, -lr);

                tensor_allocator_free(tensor_alloc, g_t);
            }
            else
            {
                // No need to clone tensor as param->grad is not modified
                // b_t <- momentum * b_t-1 + g_t
                tensor_scalar_mult_tensor_add(prev_b_t, param->grad, momentum, b_t);

                // SGD update using b_t, i.e.:
                // g_t <- b_t
                // param <- param - lr * g_t
                tensor_axpy(b_t, param, -lr);
            }
        }

        // Free and setup next iteration b_ts
        tensor_allocator_free(tensor_alloc, opt->prev_b_t[i]);
        opt->prev_b_t[i] = b_t;
    }

    return NO_ERROR;
}

void sgd_optimizer_cleanup(struct sgd_optimizer *opt)
{
    if (!opt)
    {
        return;
    }

    for (size_t i = 0; i < opt->size; i++)
    {
        tensor_allocator_free(opt->tensor_alloc, opt->prev_b_t[i]);
    }
}

static cgrad_error add_prev_b_t(struct sgd_optimizer *const state, struct tensor *const prev_grad)
{
    size_t const size = state->size;
    if (size >= MODEL_MAX_PARAMS)
    {
        return MODEL_MAX_PARAMS_EXCEEDED;
    }

    state->prev_b_t[size] = prev_grad;
    state->size++;

    return NO_ERROR;
}