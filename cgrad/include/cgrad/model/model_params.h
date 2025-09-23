#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/config.h"
#include <string.h>

/**
 * @brief Stores the parameters of a model.
 *
 * This structure contains an array of pointers to tensors representing
 * the model's learnable parameters (weights, biases, etc.) and tracks
 * the current number of parameters stored.
 */
struct model_params
{
    struct tensor *params[MODEL_MAX_PARAMS];
    size_t size;
};

/**
 * @brief Initializes a `model_params` structure.
 *
 * This function sets the size of the model parameters to 0 and
 * zero-initializes the array of parameter pointers.
 *
 * @param params Pointer to the `model_params` structure to initialize.
 *        Must not be NULL.
 *
 * @note After calling this function, the `params` array contains no valid
 *       tensor pointers and `size` is 0.
 */
void model_params_init(struct model_params *const params);

/**
 * @brief Adds a tensor to the model's parameter list.
 *
 * This function appends the given tensor to the `model_params` structure
 * and increments the size counter.
 *
 * @param params Pointer to the `model_params` structure. Must not be NULL.
 * @param t Pointer to the tensor to add as a parameter. Must not be NULL.
 *
 * @return
 * - `NO_ERROR` if the tensor was successfully added.
 * - `MODEL_MAX_PARAMS_EXCEEDED` if the parameter array has reached its maximum capacity.
 *
 * @note The function does not copy the tensor; it stores the pointer directly.
 */
cgrad_error model_params_add(struct model_params *const params, struct tensor *const t);

/**
 * @brief Zeros out the gradients of all model parameters.
 *
 * This function iterates over all tensors stored in the `model_params`
 * structure and sets all elements of their `grad` tensors to zero.
 *
 * @param params Pointer to the `model_params` structure whose gradients
 *        will be zeroed. Must not be NULL.
 *
 * @note This function assumes that each tensor in `params->params` has a
 *       valid `grad` tensor allocated and of type `double`.
 */
static inline void model_params_zero_grad(struct model_params *const params);

static inline void model_params_zero_grad(struct model_params *const params)
{
    for (size_t i = 0; i < params->size; i++)
    {
        struct tensor *grad = params->params[i]->grad;
        memset(grad->data, 0, grad->data_size * sizeof(double));
    }
}

#endif