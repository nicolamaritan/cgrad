#ifndef SGD_H
#define SGD_H

#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/model/model_params.h"
#include "cgrad/cgrad_env.h"

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer structure.
 *
 * This structure holds all information required to perform SGD updates
 * on a model's parameters, including learning rate, momentum, and optional
 * Nesterov acceleration.
 */
struct sgd_optimizer
{
    size_t size;
    struct model_params *params;
    struct tensor *prev_b_t[MODEL_MAX_PARAMS];
    struct tensor_allocator *tensor_alloc;
    double lr;
    double momemtum;
    bool nesterov;
};

/**
 * @brief Initializes a Stochastic Gradient Descent (SGD) optimizer.
 *
 * This function sets up an SGD optimizer for a given set of model parameters.
 * It allocates memory for storing previous gradients (used for momentum)
 * and configures the learning rate, momentum factor, and optional Nesterov
 * acceleration.
 *
 * @param opt Pointer to the `sgd_optimizer` structure to initialize. Must not be NULL.
 * @param params Pointer to the `model_params` structure containing the parameters
 *        to optimize. Must not be NULL.
 * @param lr Learning rate for the optimizer.
 * @param momentum Momentum factor for gradient updates (0.0 if not used).
 * @param nesterov Whether to use Nesterov accelerated gradient (true/false).
 * @param env Pointer to the CGRAD environment used for memory allocation. Must not be NULL.
 *
 * @return
 * - `NO_ERROR` if the optimizer was successfully initialized.
 * - `OPTIMIZER_NULL` if `opt` is NULL.
 * - `MODEL_PARAMS_NULL` if `params` is NULL.
 * - `CGRAD_ENV_NULL` if `env` is NULL.
 * - Any error code returned by `add_prev_b_t` or tensor allocation failures.
 *
 * @note Each parameter will have a corresponding "previous gradient" tensor
 *       allocated using the environment's tensor allocator.
 */
cgrad_error sgd_optimizer_init(struct sgd_optimizer *opt, struct model_params *const params, const double lr, const double momentum, const bool nesterov, struct cgrad_env *env);

/**
 * @brief Cleans up an SGD optimizer by freeing allocated previous gradient tensors.
 *
 * This function releases all memory allocated for the `prev_b_t` tensors
 * used to store previous gradients for momentum and Nesterov updates.
 * It does not free the `sgd_optimizer` structure itself or the model parameters.
 *
 * @param opt Pointer to the `sgd_optimizer` to clean up. If NULL, the function does nothing.
 *
 * @note After calling this function, the `prev_b_t` pointers are no longer valid.
 */
void sgd_optimizer_cleanup(struct sgd_optimizer *opt);
cgrad_error sgd_optimizer_step(struct sgd_optimizer *opt);
static inline void sgd_optimizer_zero_grad(struct sgd_optimizer *opt);

static inline void sgd_optimizer_zero_grad(struct sgd_optimizer *opt)
{
    if (!opt)
    {
        return;
    }

    model_params_zero_grad(opt->params);
}

#endif