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
#include <stdio.h>

void tensor_cpu_pool_test_init_null(struct test_result *);
void tensor_cpu_pool_test_data_alloc(struct test_result *);
void tensor_cpu_pool_test_data_free_reuse(struct test_result *);
void tensor_cpu_pool_test_exceed_size(struct test_result *);
void tensor_cpu_pool_test_free_null_safety(struct test_result *);
void tensor_cpu_pool_test_cleanup_resets(struct test_result *);
void tensor_cpu_pool_test_stress_1(struct test_result *);
void tensor_cpu_pool_test_stress_2(struct test_result *);
void tensor_cpu_pool_test_tensor_freelist(struct test_result *);
void tensor_cpu_pool_test_data_freelist(struct test_result *);
void tensor_cpu_pool_test_data_alignment(struct test_result *);

int main(int argc, char **argv)
{
    struct test_list *tests = tests_list_alloc();
    test_list_append(tests, &tensor_cpu_pool_test_init_null, "tensor_cpu_pool_test_init_null");
    test_list_append(tests, &tensor_cpu_pool_test_data_alloc, "tensor_cpu_pool_test_data_alloc");
    test_list_append(tests, &tensor_cpu_pool_test_data_free_reuse, "tensor_cpu_pool_test_data_free_reuse");
    test_list_append(tests, &tensor_cpu_pool_test_exceed_size, "tensor_cpu_pool_test_exceed_size");
    test_list_append(tests, &tensor_cpu_pool_test_free_null_safety, "tensor_cpu_pool_test_free_null_safety");
    test_list_append(tests, &tensor_cpu_pool_test_cleanup_resets, "tensor_cpu_pool_test_cleanup_resets");
    test_list_append(tests, &tensor_cpu_pool_test_stress_1, "tensor_cpu_pool_test_stress_1");
    test_list_append(tests, &tensor_cpu_pool_test_stress_2, "tensor_cpu_pool_test_stress_2");
    test_list_append(tests, &tensor_cpu_pool_test_data_alignment, "tensor_cpu_pool_test_data_alignment");
    test_list_append(tests, &tensor_cpu_pool_test_tensor_freelist, "tensor_cpu_pool_test_tensor_freelist");
    test_list_append(tests, &tensor_cpu_pool_test_data_freelist, "tensor_cpu_pool_test_data_freelist");

    run_tests(tests);

    size_t num_failed_tests = 0;
    test_list_foreach(tests, &report_failures, &num_failed_tests);

    size_t num_passed_tests = tests->size - num_failed_tests;
    float percentage_passed_tests = ((float)num_passed_tests / (float)tests->size) * 100.0;
    float percentage_failed_tests = ((float)num_failed_tests / (float)tests->size) * 100.0;

    printf("Number of tests: %ld\n", tests->size);
    printf("Number of passed tests: %ld (%.2f \%)\n", num_passed_tests, percentage_passed_tests);
    printf("Number of failed tests: %ld (%.2f \%)\n", num_failed_tests, percentage_failed_tests);

    return EXIT_SUCCESS;
}

void tensor_cpu_pool_test_init_null(struct test_result *result)
{
    cgrad_error err = tensor_cpu_pool_init(NULL);
    ASSERT_TRUE(err == MEMORY_POOL_NULL, "Expected MEMORY_POOL_NULL when initializing with NULL pool.");
test_cleanup:
    return;
}

void tensor_cpu_pool_test_data_alloc(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    cgrad_error err = tensor_cpu_pool_init(&pool);
    ASSERT_TRUE(err == NO_ERROR, "Expected NO_ERROR on init.");

    void *block1 = tensor_cpu_pool_data_alloc(&pool, 16);
    ASSERT_TRUE(block1, "data_alloc returned NULL unexpectedly.");

    void *block2 = tensor_cpu_pool_data_zero_alloc(&pool, 32);
    ASSERT_TRUE(block2, "data_zero_alloc returned NULL unexpectedly.");

    // Check zeroed memory
    char *bytes = (char *)block2;
    for (size_t i = 0; i < 32; i++)
    {
        ASSERT_TRUE(bytes[i] == 0, "Expected zeroed memory.");
    }

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_data_free_reuse(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    tensor_cpu_pool_init(&pool);

    void *block1 = tensor_cpu_pool_data_alloc(&pool, 16);
    ASSERT_TRUE(block1, "First allocation failed.");

    tensor_cpu_pool_data_free(&pool, block1);

    void *block2 = tensor_cpu_pool_data_alloc(&pool, 16);
    ASSERT_TRUE(block2, "Second allocation failed.");
    ASSERT_TRUE(block1 == block2, "Freed block should be reused.");

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_exceed_size(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    tensor_cpu_pool_init(&pool);

    void *too_large2 = tensor_cpu_pool_data_alloc(&pool, MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE + 1);
    ASSERT_TRUE(too_large2 == NULL, "Expected NULL when data alloc size exceeds limit.");

    void *too_large3 = tensor_cpu_pool_data_zero_alloc(&pool, MEMORY_TENSOR_POOL_DATA_CHUNK_SIZE + 1);
    ASSERT_TRUE(too_large3 == NULL, "Expected NULL when data_zero_alloc size exceeds limit.");

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_free_null_safety(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    tensor_cpu_pool_init(&pool);

    // Should not crash
    tensor_cpu_pool_tensor_free(&pool, NULL);
    tensor_cpu_pool_data_free(&pool, NULL);

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_cleanup_resets(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    tensor_cpu_pool_init(&pool);

    tensor_cpu_pool_cleanup(&pool);

    ASSERT_TRUE(pool.tensor_memory == NULL, "tensor_memory not reset.");
    ASSERT_TRUE(pool.data_memory == NULL, "data_memory not reset.");
    ASSERT_TRUE(pool.tensor_chunk_head == NULL, "tensor_chunk_head not reset.");
    ASSERT_TRUE(pool.data_chunk_head == NULL, "data_chunk_head not reset.");

test_cleanup:
    return;
}

void tensor_cpu_pool_test_stress_1(struct test_result *const result)
{
    struct tensor_cpu_pool pool;
    tensor_cpu_pool_init(&pool);

    struct tensor *first = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(first, "Pointer should not be null.");

    struct tensor *second = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(second, "Pointer should not be null.");

    ASSERT_TRUE(((char *)second - (char *)first == (sizeof(struct tensor) + offsetof(struct tensor_chunk, t))), "Unexpected offset.");

    tensor_cpu_pool_tensor_free(&pool, second);
    struct tensor *new_second = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(new_second, "Pointer should not be null.");
    ASSERT_TRUE((second == new_second), "Two pointers should match.");

    tensor_cpu_pool_tensor_free(&pool, first);
    struct tensor *new_first = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(new_first, "Pointer should not be null.");
    ASSERT_TRUE((first == new_first), "Two pointers should match.");

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_stress_2(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    cgrad_error err = tensor_cpu_pool_init(&pool);
    ASSERT_TRUE(err == NO_ERROR, "Expected NO_ERROR on init.");

    // --- Tensor allocations ---
    struct tensor *tensors[MEMORY_TENSOR_POOL_N_CHUNKS];

    for (size_t i = 0; i < MEMORY_TENSOR_POOL_N_CHUNKS; i++)
    {
        tensors[i] = tensor_cpu_pool_tensor_alloc(&pool);
        ASSERT_TRUE(tensors[i], "Tensor alloc failed before exhaustion.");
    }

    // Next allocation should fail
    struct tensor *extra_tensor = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(extra_tensor == NULL, "Expected NULL when pool exhausted.");

    // Free all tensors
    for (size_t i = 0; i < MEMORY_TENSOR_POOL_N_CHUNKS; i++)
    {
        tensor_cpu_pool_tensor_free(&pool, tensors[i]);
    }

    // Allocate again, should succeed
    struct tensor *t_reuse = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(t_reuse, "Failed to reuse freed tensor.");

    // --- Data allocations ---
    void *blocks[MEMORY_TENSOR_POOL_N_CHUNKS];

    for (size_t i = 0; i < MEMORY_TENSOR_POOL_N_CHUNKS; i++)
    {
        blocks[i] = tensor_cpu_pool_data_alloc(&pool, 16);
        ASSERT_TRUE(blocks[i], "Data alloc failed before exhaustion.");
    }

    // Next allocation should fail
    void *extra_block = tensor_cpu_pool_data_alloc(&pool, 16);
    ASSERT_TRUE(extra_block == NULL, "Expected NULL when data pool exhausted.");

    // Free all blocks
    for (size_t i = 0; i < MEMORY_TENSOR_POOL_N_CHUNKS; i++)
    {
        tensor_cpu_pool_data_free(&pool, blocks[i]);
    }

    // Allocate again, should succeed
    void *b_reuse = tensor_cpu_pool_data_alloc(&pool, 16);
    ASSERT_TRUE(b_reuse, "Failed to reuse freed data block.");

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_tensor_freelist(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    tensor_cpu_pool_init(&pool);

    struct tensor *t1 = tensor_cpu_pool_tensor_alloc(&pool);
    struct tensor *t2 = tensor_cpu_pool_tensor_alloc(&pool);
    struct tensor *t3 = tensor_cpu_pool_tensor_alloc(&pool);

    // Free t3 then t2, expect LIFO reuse
    tensor_cpu_pool_tensor_free(&pool, t3);
    tensor_cpu_pool_tensor_free(&pool, t2);

    struct tensor *reuse1 = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(reuse1 == t2, "Expected LIFO reuse: first should be t2.");

    struct tensor *reuse2 = tensor_cpu_pool_tensor_alloc(&pool);
    ASSERT_TRUE(reuse2 == t3, "Expected LIFO reuse: second should be t3.");

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_data_freelist(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    tensor_cpu_pool_init(&pool);

    void *b1 = tensor_cpu_pool_data_alloc(&pool, 16);
    void *b2 = tensor_cpu_pool_data_alloc(&pool, 16);
    void *b3 = tensor_cpu_pool_data_alloc(&pool, 16);

    // Free in reverse order (b3 then b2)
    tensor_cpu_pool_data_free(&pool, b3);
    tensor_cpu_pool_data_free(&pool, b2);

    void *reuse1 = tensor_cpu_pool_data_alloc(&pool, 16);
    ASSERT_TRUE(reuse1 == b2, "Expected LIFO reuse: first should be b2.");

    void *reuse2 = tensor_cpu_pool_data_alloc(&pool, 16);
    ASSERT_TRUE(reuse2 == b3, "Expected LIFO reuse: second should be b3.");

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}

void tensor_cpu_pool_test_data_alignment(struct test_result *result)
{
    struct tensor_cpu_pool pool;
    cgrad_error err = tensor_cpu_pool_init(&pool);
    ASSERT_TRUE(err == NO_ERROR, "Expected NO_ERROR on init.");

    struct data_chunk *chunk = pool.data_chunk_head;
    size_t count = 0;

    while (chunk)
    {
        uintptr_t addr = (uintptr_t)chunk->data;
        ASSERT_TRUE(addr % TENSOR_CPU_POOL_DATA_ALIGNMENT == 0,
                    "Data pointer is not 32-byte aligned.");
        chunk = chunk->next;
        count++;
    }

    ASSERT_TRUE(count == MEMORY_TENSOR_POOL_N_CHUNKS,
                "Unexpected number of data chunks traversed.");

test_cleanup:
    tensor_cpu_pool_cleanup(&pool);
}
