#include "optimizers/sgd.h"

static cgrad_error add_prev_b_t(struct sgd_optimizer *const opt, struct tensor *const prev_grad);

cgrad_error sgd_optimizer_init(struct sgd_optimizer *opt, struct model_params *const params, struct tensor_allocator *allocator)
{
    if (!opt)
    {
        return OPTIMIZER_NULL;
    }
    if (!params)
    {
        return MODEL_PARAMS_NULL;
    }
    if (!allocator)
    {
        return TENSOR_ALLOCATOR_NULL;
    }

    opt->params = params;
    opt->allocator = allocator;
    opt->size = 0;
    for (size_t i = 0; i < params->size; i++)
    {
        struct tensor* param = params->params[i];
        struct tensor* param_prev_grad = tensor_allocator_no_grad_zero_alloc(allocator, param->shape, param->shape_size);

        cgrad_error err = add_prev_b_t(opt, param_prev_grad);
        if (err != NO_ERROR)
        {
            return err;
        }
    }

    return NO_ERROR;
}

cgrad_error sgd_optimizer_step(struct sgd_optimizer* opt, double lr, double momentum, bool nesterov)
{
    if (!opt)
    {
        return OPTIMIZER_NULL;
    }

    for (size_t i = 0; i < opt->params->size; i++)
    {
        struct tensor* param = opt->params->params[i];
        struct tensor_allocator *allocator = opt->allocator;
        
        struct tensor* g_t = tensor_allocator_clone(allocator, param->grad);
        struct tensor* prev_b_t = opt->prev_b_t[i];
        struct tensor* b_t = tensor_allocator_clone(allocator, prev_b_t);
        size_t grad_size = g_t->data_size;

        if (momentum != 0)
        {
            // b_t <- momentum * b_t-1 + g_t
            for (size_t j = 0; j < grad_size; j++)
            {
                b_t->data[j] = momentum * prev_b_t->data[j] + g_t->data[j];
            }

            if (nesterov)
            {
                // g_t <- g_t + momentum * b_t
                for (size_t j = 0; j < grad_size; j++)
                {
                    g_t->data[j] += (momentum * b_t->data[j]);
                }
            }
            else
            {
                // g_t <- b_t
                tensor_copy(b_t, g_t);
            }
        }

        // SGD update using g_t
        double* g_t_data = g_t->data;
        for (size_t i = 0; i < grad_size; i++)
        {
            g_t_data[i] *= -lr;
        }

        tensor_add_inplace(param, g_t);

        // Free and setup next iteration b_ts
        tensor_allocator_free(allocator, g_t);
        tensor_allocator_free(allocator, opt->prev_b_t[i]);
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
        tensor_allocator_free(opt->allocator, opt->prev_b_t[i]);
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