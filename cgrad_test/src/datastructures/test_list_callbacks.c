#include "cgrad_test/datastructures/test_list_callbacks.h"
#include <stdio.h>

void report_failures(const char *test_name, struct test_result *result, void *user)
{
    size_t *num_failed_tests = user;

    if (result->err == TEST_FAILED)
    {
        printf("\t - Test \"%s\" failed.\n", test_name);
        printf("\t   Message: %s.\n", result->msg);
        *num_failed_tests++;
    }
}