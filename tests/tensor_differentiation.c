#include "cgrad_test/assert.h"
#include "cgrad_test/config.h"
#include "cgrad_test/test_result.h"
#include "cgrad_test/test_case.h"
#include "cgrad_test/datastructures/test_list/test_list.h"
#include "cgrad_test/datastructures/test_list/test_list_callbacks.h"
#include "cgrad_test/run_tests.h"
#include "cgrad_test/utils/sum_loss.h"
#include "cgrad/tensor/tensor_alloc.h"
#include "cgrad/tensor/tensor_equality.h"
#include "cgrad/tensor/tensor_set.h"
#include "cgrad/tensor/tensor_const_scalar_mult.h"
#include "cgrad/tensor/tensor_softmax_last_axis.h"
#include "cgrad/tensor/tensor_reshape.h"
#include "cgrad/tensor/tensor2d_mult.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_allocator.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include <stdio.h>

/**
 * @brief Tests the backward pass of tensor_const_scalar_mult through the
 *        real autodiff engine, composing it with the sum_loss toy loss
 *        so the incoming gradient at its output is a tensor of all 1s.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_const_scalar_mult_backward_test_cpu_instance_1(struct test_result *result);

/**
 * @brief Same as instance_1, but with a negative scalar, to check the
 *        sign is propagated correctly into the gradient.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_const_scalar_mult_backward_test_cpu_instance_2(struct test_result *result);

/**
 * @brief Tests the backward pass of tensor_reshape through the real
 *        autodiff engine. Since reshape does not alter values, composing
 *        it with sum_loss means the gradient w.r.t. the original tensor
 *        must be all 1s, reshaped back to the ORIGINAL shape.
 */
void tensor_reshape_backward_test_cpu_instance_1(struct test_result *result);

/**
 * @brief Tests the backward pass of softmax through the real autodiff
 *        engine, composed with sum_loss. Since every softmax row always
 *        sums to the constant 1, its gradient with respect to the
 *        original logits must be exactly 0, regardless of the input
 *        values - a strong, easy-to-verify-by-hand sanity check.
 */
void tensor_softmax_last_axis_backward_test_cpu_instance_1(struct test_result *result);

/**
 * @brief Tests the backward pass of softmax on a non-degenerate case:
 *        composes softmax with a matrix multiplication against a fixed,
 *        non-uniform weight vector before reducing via sum_loss. Unlike
 *        composing sum_loss directly with softmax (which always yields
 *        a zero gradient, since every softmax row sums to the constant
 *        1), this exercises the actual Jacobian-vector product formula
 *        with a grad_wrt_out that varies across the last axis.
 */
void tensor_softmax_last_axis_backward_test_cpu_instance_2(struct test_result *result);

int main(int argc, char **argv)
{
    struct test_list *tests = tests_list_alloc();
    test_list_append(tests, &tensor_const_scalar_mult_backward_test_cpu_instance_1, "tensor_const_scalar_mult_backward_test_cpu_instance_1");
    test_list_append(tests, &tensor_const_scalar_mult_backward_test_cpu_instance_2, "tensor_const_scalar_mult_backward_test_cpu_instance_2");
    test_list_append(tests, &tensor_reshape_backward_test_cpu_instance_1, "tensor_reshape_backward_test_cpu_instance_1");
    test_list_append(tests, &tensor_softmax_last_axis_backward_test_cpu_instance_1, "tensor_softmax_last_axis_backward_test_cpu_instance_1");
    test_list_append(tests, &tensor_softmax_last_axis_backward_test_cpu_instance_2, "tensor_softmax_last_axis_backward_test_cpu_instance_2");

    run_tests(tests);

    size_t num_failed_tests = 0;
    test_list_foreach(tests, &report_failures, &num_failed_tests);

    size_t num_passed_tests = tests->size - num_failed_tests;
    float percentage_passed_tests = ((float)num_passed_tests / (float)tests->size) * 100.0;
    float percentage_failed_tests = ((float)num_failed_tests / (float)tests->size) * 100.0;

    printf("Number of tests: %ld\n", tests->size);
    printf("Number of passed tests: %ld (%.2f %%)\n", num_passed_tests, percentage_passed_tests);
    printf("Number of failed tests: %ld (%.2f %%)\n", num_failed_tests, percentage_failed_tests);

    return EXIT_SUCCESS;
}

void tensor_const_scalar_mult_backward_test_cpu_instance_1(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 2};
    const float t_data[] = {1.0, 2.0, 3.0, 4.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    const double scalar = 3.0;

    // Forward: out = t * scalar
    struct tensor *out = NULL;
    cgrad_error err = tensor_const_scalar_mult(t, scalar, &out, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "tensor_const_scalar_mult forward should not fail.");

    struct tensor *z = NULL;
    err = sum_loss(out, &z, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "sum_loss forward should not fail.");

    backward(z, &env);

    const float expected_grad_data[] = {3.0, 3.0, 3.0, 3.0};
    struct tensor *expected_grad = tensor_from_array_alloc(&env, expected_grad_data, shape, 2, DTYPE);

    ASSERT_TRUE(tensor_no_grad_equal(t->grad, expected_grad), "Gradient with respect to t is incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_const_scalar_mult_backward_test_cpu_instance_2(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {3, 1};
    const float t_data[] = {1.0, -2.0, 5.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    const double scalar = -2.5;

    struct tensor *out = NULL;
    cgrad_error err = tensor_const_scalar_mult(t, scalar, &out, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "tensor_const_scalar_mult forward should not fail.");

    struct tensor *z = NULL;
    err = sum_loss(out, &z, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "sum_loss forward should not fail.");

    backward(z, &env);

    // Every element of the gradient must equal the scalar, regardless of
    // the corresponding value of t, since d(t_i * scalar)/dt_i = scalar
    const float expected_grad_data[] = {-2.5, -2.5, -2.5};
    struct tensor *expected_grad = tensor_from_array_alloc(&env, expected_grad_data, shape, 2, DTYPE);

    ASSERT_TRUE(tensor_no_grad_equal(t->grad, expected_grad), "Gradient with respect to t is incorrect for a negative scalar.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_reshape_backward_test_cpu_instance_1(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 3};
    const float t_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    const size_t new_shape[] = {3, 2};

    struct tensor *out = NULL;
    cgrad_error err = tensor_reshape(t, new_shape, 2, &out, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "tensor_reshape forward should not fail.");

    struct tensor *z = NULL;
    err = sum_loss(out, &z, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "sum_loss forward should not fail.");

    backward(z, &env);

    // Reshape does not alter values, so dz/dt_i = 1 for every original
    // element, and the gradient must come back with t's ORIGINAL shape
    const float expected_grad_data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    struct tensor *expected_grad = tensor_from_array_alloc(&env, expected_grad_data, shape, 2, DTYPE);

    ASSERT_TRUE(tensor_no_grad_equal(t->grad, expected_grad), "Gradient with respect to t is incorrect, or was not reshaped back to the original shape.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_backward_test_cpu_instance_1(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 3};
    // Arbitrary values are fine here: the property being tested holds
    // regardless of the specific numbers
    const float t_data[] = {0.5, -2.3, 7.1, -1.0, 1.0, 0.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_softmax_last_axis(t, &out, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "Softmax forward should not fail.");

    struct tensor *z = NULL;
    err = sum_loss(out, &z, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "sum_loss forward should not fail.");

    backward(z, &env);

    // sum_j(softmax(t)_j) == 1 for every sample, always: it's a constant
    // function of t, so its gradient with respect to t must be exactly 0
    const float expected_grad_data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    struct tensor *expected_grad = tensor_from_array_alloc(&env, expected_grad_data, shape, 2, DTYPE);

    ASSERT_TRUE(tensor_no_grad_equal(t->grad, expected_grad), "Gradient of sum(softmax(t)) w.r.t. t must be exactly zero.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_backward_test_cpu_instance_2(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 3};
    const float t_data[] = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    // Fixed, non-uniform weight vector: isolates y_0 of each row as h_i,
    // so grad_wrt_out varies across the last axis instead of being constant
    const size_t w_shape[] = {3, 1};
    const float w_data[] = {1.0, 0.0, 0.0};
    struct tensor *w = tensor_from_array_alloc(&env, w_data, w_shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_softmax_last_axis(t, &out, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "Softmax forward should not fail.");

    struct tensor *h = NULL;
    err = tensor2d_mult(out, w, &h, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "tensor2d_mult forward should not fail.");

    struct tensor *z = NULL;
    err = sum_loss(h, &z, true, &env);
    ASSERT_TRUE(err == NO_ERROR, "sum_loss forward should not fail.");

    backward(z, &env);

    const float expected_grad_data[] = {
        0.081925f, -0.022033f, -0.059892f,
        0.222222f, -0.111111f, -0.111111f
    };
    struct tensor *expected_grad = tensor_from_array_alloc(&env, expected_grad_data, shape, 2, DTYPE);

    ASSERT_TRUE(tensor_no_grad_equal(t->grad, expected_grad), "Gradient of softmax w.r.t. t is incorrect on non-degenerate input.");

test_cleanup:
    cgrad_env_cleanup(&env);
}