#include "cgrad_test/assert.h"
#include "cgrad_test/config.h"
#include "cgrad_test/test_result.h"
#include "cgrad_test/test_case.h"
#include "cgrad_test/datastructures/test_list/test_list.h"
#include "cgrad_test/datastructures/test_list/test_list_callbacks.h"
#include "cgrad_test/run_tests.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_allocator.h"
#include "cgrad/tensor/tensor_set.h"
#include "cgrad/tensor/tensor2d_mult.h"
#include "cgrad/tensor/tensor_add.h"
#include "cgrad/tensor/tensor_equality.h"
#include "cgrad/tensor/tensor_alloc.h"
#include "cgrad/tensor/tensor_softmax_last_axis.h"
#include "cgrad/tensor/tensor_const_scalar_mult.h"
#include <stdio.h>
#include <math.h>

/**
 * @brief Implements a first test for the multiplication between 2D tensors 
 * 
 * @param result Pointer to a test_result struct where to save the results
 * 
 * @return None
 */
void tensor2d_mult_test_cpu_instance_1(struct test_result *result);

/**
 * @brief Implements a first test for the sum between 2D tensors 
 * 
 * @param result Pointer to a test_result struct where to save the results
 * 
 * @return None
 */
void tensor_add_test_cpu_instance_1(struct test_result *result);

/**
 * @brief Implements a second test for the sum between 2D tensors
 * 
 * @param result Pointer to a test_result struct where to save the results
 * 
 * @return None
 */
void tensor_add_test_cpu_instance_2(struct test_result *result);

/**
 * @brief Implements a third test for the sum between 2D tensors
 * 
 * @param result Pointer to a test_result struct where to save the results
 * 
 * @return None
 */
void tensor_add_test_cpu_instance_3(struct test_result *result);

/**
 * @brief Implements a fourth test for the sum between 2D tensors
 * 
 * @param result Pointer to a test_result struct where to save the results
 * 
 * @return None
 */
void tensor_add_test_cpu_instance_4(struct test_result *result);

/**
 * @brief Implements a test for the sum between 2D tensors
 * 
 * @param result Pointer to a test_result struct where to save the results
 * 
 * @return None
 */
void tensor_add_test_cpu_instance_5(struct test_result *result);

/**
 * @brief Tests softmax along the last axis on a basic case with small
 *        values (2x3, float32).
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_softmax_last_axis_test_cpu_instance_1(struct test_result *result);

/**
 * @brief Tests softmax along the last axis with extreme values (float64),
 *        verifying numerical stability against overflow/underflow.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_softmax_last_axis_test_cpu_instance_2(struct test_result *result);

/**
 * @brief Tests softmax on an unsupported dtype, expecting an error.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_softmax_last_axis_test_cpu_instance_3(struct test_result *result);

/**
 * @brief Verifies that every row of the softmax output sums to 1 and
 *        that each value is a valid probability (property-based test,
 *        not tied to hardcoded expected values).
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_softmax_last_axis_test_cpu_instance_4(struct test_result *result);

/**
 * @brief Tests softmax along the last axis on a rank-3 tensor (batch of
 *        sequences), verifying that softmax is applied independently
 *        per sample across the two leading dimensions.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_softmax_last_axis_test_cpu_instance_5(struct test_result *result);

/**
 * @brief Tests softmax along the last axis on a rank-4 tensor, verifying
 *        the row-sums-to-1 property holds across all leading dimensions
 *        collapsed by get_samples_number.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_softmax_last_axis_test_cpu_instance_6(struct test_result *result);

/**
 * @brief Tests tensor_const_scalar_mult with a basic positive scalar
 *        (2x2, float32).
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_const_scalar_mult_test_cpu_instance_1(struct test_result *result);

/**
 * @brief Tests tensor_const_scalar_mult with a negative scalar (float64).
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_const_scalar_mult_test_cpu_instance_2(struct test_result *result);

/**
 * @brief Tests tensor_const_scalar_mult with a zero scalar, expecting an
 *        all-zero output regardless of the input values.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_const_scalar_mult_test_cpu_instance_3(struct test_result *result);

/**
 * @brief Tests tensor_const_scalar_mult on an unsupported dtype,
 *        expecting an error.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_const_scalar_mult_test_cpu_instance_4(struct test_result *result);

/**
 * @brief Tests tensor_const_scalar_mult on a rank-3 tensor, verifying
 *        the scalar is applied elementwise across every leading
 *        dimension, not just the last one.
 *
 * @param result Pointer to a test_result struct where to save the results
 *
 * @return None
 */
void tensor_const_scalar_mult_test_cpu_instance_5(struct test_result *result);

int main(int argc, char **argv)
{
    struct test_list *tests = tests_list_alloc();
    test_list_append(tests, &tensor2d_mult_test_cpu_instance_1, "tensor2d_mult_test_cpu_instance_1");
    test_list_append(tests, &tensor_add_test_cpu_instance_1, "tensor_add_test_cpu_instance_1");
    test_list_append(tests, &tensor_add_test_cpu_instance_2, "tensor_add_test_cpu_instance_2");
    test_list_append(tests, &tensor_add_test_cpu_instance_3, "tensor_add_test_cpu_instance_3");
    test_list_append(tests, &tensor_add_test_cpu_instance_4, "tensor_add_test_cpu_instance_4");
    test_list_append(tests, &tensor_add_test_cpu_instance_5, "tensor_add_test_cpu_instance_5");
    test_list_append(tests, &tensor_softmax_last_axis_test_cpu_instance_1, "tensor_softmax_last_axis_test_cpu_instance_1");
    test_list_append(tests, &tensor_softmax_last_axis_test_cpu_instance_2, "tensor_softmax_last_axis_test_cpu_instance_2");
    test_list_append(tests, &tensor_softmax_last_axis_test_cpu_instance_3, "tensor_softmax_last_axis_test_cpu_instance_3");
    test_list_append(tests, &tensor_softmax_last_axis_test_cpu_instance_4, "tensor_softmax_last_axis_test_cpu_instance_4");
    test_list_append(tests, &tensor_softmax_last_axis_test_cpu_instance_5, "tensor_softmax_last_axis_test_cpu_instance_5");
    test_list_append(tests, &tensor_softmax_last_axis_test_cpu_instance_6, "tensor_softmax_last_axis_test_cpu_instance_6");
    test_list_append(tests, &tensor_const_scalar_mult_test_cpu_instance_1, "tensor_const_scalar_mult_test_cpu_instance_1");
    test_list_append(tests, &tensor_const_scalar_mult_test_cpu_instance_2, "tensor_const_scalar_mult_test_cpu_instance_2");
    test_list_append(tests, &tensor_const_scalar_mult_test_cpu_instance_3, "tensor_const_scalar_mult_test_cpu_instance_3");
    test_list_append(tests, &tensor_const_scalar_mult_test_cpu_instance_4, "tensor_const_scalar_mult_test_cpu_instance_4");
    test_list_append(tests, &tensor_const_scalar_mult_test_cpu_instance_5, "tensor_const_scalar_mult_test_cpu_instance_5");

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

void tensor2d_mult_test_cpu_instance_1(struct test_result *const result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 2};
    const float t1_data[] = {1.0, 2.0, 3.0, 4.0};
    struct tensor *t1 = tensor_from_array_alloc(&env, t1_data, shape, 2, DTYPE);

    const float t2_data[] = {1.0, 2.0, 3.0, 4.0};
    struct tensor *t2 = tensor_from_array_alloc(&env, t2_data, shape, 2, DTYPE);

    const float expected_out_data[] = {7.0, 10.0, 15.0, 22.0};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    tensor2d_mult(t1, t2, &out, false, &env);

    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_add_test_cpu_instance_1(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 2};
    const float t1_data[] = {1.0, 2.0, 3.0, 4.0};
    struct tensor *t1 = tensor_from_array_alloc(&env, t1_data, shape, 2, DTYPE);

    const float t2_data[] = {5.0, 6.0, 7.0, 8.0};
    struct tensor *t2 = tensor_from_array_alloc(&env, t2_data, shape, 2, DTYPE);

    const float expected_out_data[] = {6.0, 8.0, 10.0, 12.0};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    tensor_add(t1, t2, &out, false, &env);

    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_add_test_cpu_instance_2(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {3, 2};
    const float t1_data[] = {0.1, 0.2, 1.0, 1.1, -2.0, 0.5};
    struct tensor *t1 = tensor_from_array_alloc(&env, t1_data, shape, 2, DTYPE);

    const float t2_data[] = {0.0, 2.0, -1.0, 10.0, 7.0, 8.0};
    struct tensor *t2 = tensor_from_array_alloc(&env, t2_data, shape, 2, DTYPE);

    const float expected_out_data[] = {0.1, 2.2, 0.0, 11.1, 5.0, 8.5};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    tensor_add(t1, t2, &out, false, &env);

    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_add_test_cpu_instance_3(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {3, 2};
    const float t1_data[] = {0.1, 0.2, 1.0, 1.1, -2.0, 0.5};
    struct tensor *t1 = tensor_from_array_alloc(&env, t1_data, shape, 2, DTYPE);

    const float t2_data[] = {0.0, 2.0, -1.0, 10.0, 7.0, 8.0};
    struct tensor *t2 = tensor_from_array_alloc(&env, t2_data, shape, 2, DTYPE);

    const float expected_out_data[] = {1.1, 1.2, 1.0, 10.1, 7.0, 8.5};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    tensor_add(t1, t2, &out, false, &env);

    ASSERT_FALSE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_add_test_cpu_instance_4(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_INT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {3, 2};
    const int32_t t1_data[] = {1, 2, 1, 3, -2, 5};
    //Il DTYPE qui dentro serve per fare calcoli e settare il campo dtype sul tensore
    struct tensor *t1 = tensor_from_array_alloc(&env, t1_data, shape, 2, DTYPE);

    const int32_t t2_data[] = {0, 2, -1, 10, 14, 8};
    struct tensor *t2 = tensor_from_array_alloc(&env, t2_data, shape, 2, DTYPE);

    const int32_t expected_out_data[] = {1, 4, 0, 13, 12, 13};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    tensor_add(t1, t2, &out, false, &env);

    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_add_test_cpu_instance_5(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_INT16;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {3, 2};
    const int16_t t1_data[] = {3, 126, -12, 34, -2, 5};
    //Il DTYPE qui dentro serve per fare calcoli e settare il campo dtype sul tensore
    struct tensor *t1 = tensor_from_array_alloc(&env, t1_data, shape, 2, DTYPE);

    const int16_t t2_data[] = {0, 2, -1, 10, 14, 8};
    struct tensor *t2 = tensor_from_array_alloc(&env, t2_data, shape, 2, DTYPE);

    const int16_t expected_out_data[] = {3, 128, -13, 44, 12, 13};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    tensor_add(t1, t2, &out, false, &env);

    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_test_cpu_instance_1(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 3};
    const float t_data[] = {1.0, 2.0, 3.0, 1.0, 1.0, 1.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    // Row 1: softmax(1,2,3); row 2: softmax(1,1,1) = uniform distribution
    const float expected_out_data[] = {0.090031f, 0.244728f, 0.665241f, 0.333333f, 0.333333f, 0.333333f};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    tensor_softmax_last_axis(t, &out, false, &env);

    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_test_cpu_instance_2(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT64;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 2};
    // Row 1: equal values; row 2: extreme values that would overflow
    // with a naive exp(max) implementation
    const double t_data[] = {0.0, 0.0, -1000.0, 1000.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    // Row 1: softmax(0,0) = [0.5, 0.5]
    // Row 2: softmax(-1000,1000), the second class completely dominates -> [0.0, 1.0]
    const double expected_out_data[] = {0.5, 0.5, 0.0, 1.0};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_softmax_last_axis(t, &out, false, &env);

    ASSERT_TRUE(err == NO_ERROR, "Softmax should not fail on extreme values.");
    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect (possible overflow/underflow bug).");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_test_cpu_instance_3(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_INT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 2};
    const int32_t t_data[] = {1, 2, 3, 4};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_softmax_last_axis(t, &out, false, &env);

    ASSERT_TRUE(err == OPERATION_INVALID_TENSOR_DTYPE, "Softmax on integer dtype should return OPERATION_INVALID_TENSOR_DTYPE.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_test_cpu_instance_4(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT64;
    const double EPSILON = 1e-9;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {3, 4};
    const double t_data[] = {
        0.5, -2.3, 7.1, 0.0,
        -5.0, -5.0, -5.0, -5.0,
        100.0, 99.0, 98.0, 97.0
    };
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_softmax_last_axis(t, &out, false, &env);
    ASSERT_TRUE(err == NO_ERROR, "Softmax should not fail.");

    double *out_data = (double *)out->data;
    size_t num_samples = shape[0];
    size_t sample_size = shape[1];

    for (size_t i = 0; i < num_samples; i++)
    {
        double row_sum = 0.0;
        for (size_t j = 0; j < sample_size; j++)
        {
            double val = out_data[i * sample_size + j];
            ASSERT_TRUE(val >= 0.0 && val <= 1.0, "Softmax output must be a valid probability in [0, 1].");
            row_sum += val;
        }
        ASSERT_TRUE(fabs(row_sum - 1.0) < EPSILON, "Each softmax row must sum to 1.");
    }

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_test_cpu_instance_5(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    // Shape (2, 2, 3): 2 batches, 2 timesteps, 3 classes -> 4 independent samples of size 3
    const size_t shape[] = {2, 2, 3};
    const float t_data[] = {
        1.0, 2.0, 3.0,   // batch 0, timestep 0
        1.0, 1.0, 1.0,   // batch 0, timestep 1
        0.0, 0.0, 0.0,   // batch 1, timestep 0
        3.0, 2.0, 1.0    // batch 1, timestep 1
    };
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 3, DTYPE);

    const float expected_out_data[] = {
        0.090031f, 0.244728f, 0.665241f,
        0.333333f, 0.333333f, 0.333333f,
        0.333333f, 0.333333f, 0.333333f,
        0.665241f, 0.244728f, 0.090031f
    };
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 3, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_softmax_last_axis(t, &out, false, &env);

    ASSERT_TRUE(err == NO_ERROR, "Softmax should not fail on a rank-3 tensor.");
    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect on rank-3 input.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_softmax_last_axis_test_cpu_instance_6(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT64;
    const double EPSILON = 1e-9;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    // Shape (2, 2, 2, 3): 8 independent samples of size 3, exercising
    // get_samples_number's collapse of 3 leading dimensions
    const size_t shape[] = {2, 2, 2, 3};
    const double t_data[] = {
        0.5, -2.3, 7.1,
        -5.0, -5.0, -5.0,
        100.0, 99.0, 98.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 3.0,
        -1.0, 1.0, 0.0,
        10.0, -10.0, 0.0,
        3.3, 3.3, 3.3
    };
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 4, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_softmax_last_axis(t, &out, false, &env);
    ASSERT_TRUE(err == NO_ERROR, "Softmax should not fail on a rank-4 tensor.");

    double *out_data = (double *)out->data;
    size_t num_samples = shape[0] * shape[1] * shape[2]; // 8, collapsing all leading dims
    size_t sample_size = shape[3]; // 3

    for (size_t i = 0; i < num_samples; i++)
    {
        double row_sum = 0.0;
        for (size_t j = 0; j < sample_size; j++)
        {
            double val = out_data[i * sample_size + j];
            ASSERT_TRUE(val >= 0.0 && val <= 1.0, "Softmax output must be a valid probability in [0, 1].");
            row_sum += val;
        }
        ASSERT_TRUE(fabs(row_sum - 1.0) < EPSILON, "Each softmax row must sum to 1, even on rank-4 input.");
    }

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_const_scalar_mult_test_cpu_instance_1(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 2};
    const float t_data[] = {1.0, 2.0, 3.0, 4.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    const double scalar = 2.0;
    const float expected_out_data[] = {2.0, 4.0, 6.0, 8.0};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_const_scalar_mult(t, scalar, &out, false, &env);

    ASSERT_TRUE(err == NO_ERROR, "tensor_const_scalar_mult should not fail.");
    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_const_scalar_mult_test_cpu_instance_2(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT64;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {3, 2};
    const double t_data[] = {1.0, -2.0, 3.5, 0.0, -1.0, 4.2};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    const double scalar = -1.5;
    const double expected_out_data[] = {-1.5, 3.0, -5.25, 0.0, 1.5, -6.3};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_const_scalar_mult(t, scalar, &out, false, &env);

    ASSERT_TRUE(err == NO_ERROR, "tensor_const_scalar_mult should not fail.");
    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect with a negative scalar.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_const_scalar_mult_test_cpu_instance_3(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 3};
    const float t_data[] = {1.0, 2.0, 3.0, -4.0, 5.5, -6.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    const double scalar = 0.0;
    const float expected_out_data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 2, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_const_scalar_mult(t, scalar, &out, false, &env);

    ASSERT_TRUE(err == NO_ERROR, "tensor_const_scalar_mult should not fail.");
    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "Multiplying by zero should zero out every element.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_const_scalar_mult_test_cpu_instance_4(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_INT32;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    const size_t shape[] = {2, 2};
    const int32_t t_data[] = {1, 2, 3, 4};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 2, DTYPE);

    const double scalar = 2.0;
    struct tensor *out = NULL;
    cgrad_error err = tensor_const_scalar_mult(t, scalar, &out, false, &env);

    ASSERT_TRUE(err == OPERATION_INVALID_TENSOR_DTYPE, "tensor_const_scalar_mult on integer dtype should return OPERATION_INVALID_TENSOR_DTYPE.");

test_cleanup:
    cgrad_env_cleanup(&env);
}

void tensor_const_scalar_mult_test_cpu_instance_5(struct test_result *result)
{
    const int SEED = 42;
    const size_t INTERMEDIATES_CAPACITY = 20;
    const cgrad_dtype DTYPE = DTYPE_FLOAT64;

    struct cgrad_env env;
    ASSERT_TRUE(cgrad_env_init(&env, SEED, INTERMEDIATES_CAPACITY) == NO_ERROR, "CGrad Environment Initialization should not fail.");

    // Shape (2, 2, 2): the scalar must be applied to every element,
    // regardless of which leading dimension it belongs to
    const size_t shape[] = {2, 2, 2};
    const double t_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    struct tensor *t = tensor_from_array_alloc(&env, t_data, shape, 3, DTYPE);

    const double scalar = 3.0;
    const double expected_out_data[] = {3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0};
    struct tensor *expected_out = tensor_from_array_alloc(&env, expected_out_data, shape, 3, DTYPE);

    struct tensor *out = NULL;
    cgrad_error err = tensor_const_scalar_mult(t, scalar, &out, false, &env);

    ASSERT_TRUE(err == NO_ERROR, "tensor_const_scalar_mult should not fail on a rank-3 tensor.");
    ASSERT_TRUE(tensor_no_grad_equal(out, expected_out), "One or more output values incorrect on rank-3 input.");

test_cleanup:
    cgrad_env_cleanup(&env);
}