#include "cgrad_test/tests/tensor/tensor2d_mult_tests.h"
#include "cgrad_test/test_result_error.h"
#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include "cgrad/memory/computational_graph/computational_graph_cpu_allocator.h"
#include "cgrad/tensor/tensor_set.h"
#include "cgrad/tensor/tensor2d_mult.h"
#include "cgrad/tensor/tensor_equality.h"

void tensor2d_mult_test_cpu_instance_1(struct test_result *const result)
{
    struct tensor_allocator tensor_alloc;
    tensor_cpu_allocator_init(&tensor_alloc);

    struct computational_graph_allocator graph_alloc;
    computational_graph_cpu_allocator_init(&graph_alloc);

    struct allocators allocs = {&tensor_alloc, &graph_alloc};

    const cgrad_dtype DTYPE = DTYPE_FLOAT32;

    size_t shape[] = {2, 2};
    struct tensor *t1 = tensor_allocator_alloc(&tensor_alloc, shape, 2, DTYPE);
    tensor2d_set(t1, 0, 0, 1.0f);
    tensor2d_set(t1, 0, 1, 2.0f);
    tensor2d_set(t1, 1, 0, 3.0f);
    tensor2d_set(t1, 1, 1, 4.0f);

    struct tensor *t2 = tensor_allocator_alloc(&tensor_alloc, shape, 2, DTYPE);
    tensor2d_set(t2, 0, 0, 1.0f);
    tensor2d_set(t2, 0, 1, 2.0f);
    tensor2d_set(t2, 1, 0, 3.0f);
    tensor2d_set(t2, 1, 1, 4.0f);

    struct tensor *EXPECTED_OUT = tensor_allocator_alloc(&tensor_alloc, shape, 2, DTYPE);
    tensor2d_set(EXPECTED_OUT, 0, 0, 7.0f);
    tensor2d_set(EXPECTED_OUT, 0, 1, 10.0f);
    tensor2d_set(EXPECTED_OUT, 1, 0, 15.0f);
    tensor2d_set(EXPECTED_OUT, 1, 1, 22.0f);

    struct tensor *out = NULL;
    tensor2d_mult(t1, t2, &out, false, &allocs);

    if (!tensor_no_grad_equal(out, EXPECTED_OUT))
    {
        test_result_set_error(result, TEST_FAILED, "One or more output values incorrect.");
    }
}