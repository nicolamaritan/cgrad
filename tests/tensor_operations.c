#include "cgrad_test/config.h"
#include "cgrad_test/test_result.h"
#include "cgrad_test/test_case.h"
#include "cgrad_test/datastructures/test_list.h"
#include "cgrad_test/tests/tensor/tensor_tests.h"
#include "cgrad_test/run_tests.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    struct test_list *tests = tests_list_alloc();
    test_list_append(tests, &tensor2d_mult_test_cpu_instance_1, "tensor2d_mult_test_cpu_instance_1");

    run_tests(tests); 

    size_t num_failed_tests = 0;
    struct test_list_node *curr = tests->head->next;
    while (curr != tests->tail)
    {
        if (curr->result.err == TEST_FAILED)
        {
            printf("\t - Test \"%s\" failed.\n", curr->test_name);
            printf("\t   Message: %s.\n", curr->result.msg);
            num_failed_tests++;
        }

        curr = curr->next;
    }

    size_t num_passed_tests = tests->size - num_failed_tests; 
    float percentage_passed_tests = ((float)num_passed_tests / (float)tests->size) * 100.0;
    
    printf("Number of passed tests: %ld (%.2f \%)\n", num_passed_tests, percentage_passed_tests);

    return EXIT_SUCCESS;
}