#include "cgrad_test/datastructures/test_list.h"
#include <string.h>

void test_list_free(struct test_list *const list)
{
    if (!list)
    {
        return;
    }

    struct test_list_node *curr = list->head;
    while (curr && curr != list->tail)
    {
        struct test_list_node *next = curr->next;
        test_list_node_free(curr);
        curr = next;
    }

    test_list_node_free(list->tail);
    free(list);
}

void test_list_append(struct test_list *const list, test_case test_case_func, const char *test_name)
{
    if (!list)
    {
        return;
    }
    if (!test_case_func)
    {
        return;
    }

    struct test_list_node *node = calloc(1, sizeof(struct test_list_node));
    node->test_case_func = test_case_func;

    const char NULL_TERMINATOR = 0;
    memset(node->test_name, 0, sizeof(node->test_name));
    memccpy(node->test_name, test_name, NULL_TERMINATOR, sizeof(node->test_name) - 1);

    node->next = list->tail;
    node->prev = list->tail->prev;
    list->tail->prev->next = node;
    list->tail->prev = node;

    list->size++;
}