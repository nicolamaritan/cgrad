#include "optimizers/sgd.h"

cgrad_error add_prev_b_t(sgd_state *const state, struct tensor *const prev_grad);

void sgd_step(double lr, double momentum, bool nesterov, sgd_state* state, model_params* params)
{
    for (size_t i = 0; i < params->size; i++)
    {
        struct tensor* param = params->params[i];
        
        struct tensor* g_t = tensor_clone(param->grad);
        struct tensor* prev_b_t = state->prev_b_t[i];
        struct tensor* b_t = tensor_clone(prev_b_t);
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
        tensor_free(g_t);
        tensor_free(state->prev_b_t[i]);
        state->prev_b_t[i] = b_t;
    }
}

cgrad_error init_sgd_state(sgd_state *state, const model_params *const params)
{
    state->size = 0;
    for (size_t i = 0; i < params->size; i++)
    {
        struct tensor* param = params->params[i];
        struct tensor* param_prev_grad = tensor_no_grad_zero_alloc(param->shape, param->shape_size);

        cgrad_error err = add_prev_b_t(state, param_prev_grad);
        if (err != NO_ERROR)
        {
            return err;
        }
    }

    return NO_ERROR;
}

void free_sgd_state_tensors(sgd_state *state)
{
    for (size_t i = 0; i < state->size; i++)
    {
        tensor_free(state->prev_b_t[i]);
    }
}

cgrad_error add_prev_b_t(sgd_state *const state, struct tensor *const prev_grad)
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