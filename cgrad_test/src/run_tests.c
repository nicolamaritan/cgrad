#include "cgrad_test/run_tests.h"

void run_tests(struct test_list *const tests)
{
    struct test_list_node *curr = tests->head->next;
    while (curr != tests->tail)
    {
        curr->test_case_func(&curr->result);
        curr = curr->next;
    }
}