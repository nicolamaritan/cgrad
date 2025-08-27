#include "cgrad_test/datastructures/test_list/test_list.h"
#include <string.h>

static struct test_list_node *test_list_node_alloc(test_case test_case_func);
static void test_list_node_free(struct test_list_node *const node);

struct test_list *tests_list_alloc()
{
    struct test_list *list = calloc(1, sizeof(struct test_list));
    if (!list)
    {
        return NULL;
    }

    list->head = test_list_node_alloc(NULL);
    if (!list->head)
    {
        test_list_free(list);
    }

    list->tail = test_list_node_alloc(NULL);
    if (!list->tail)
    {
        test_list_node_free(list->head);
        test_list_free(list);
    }

    list->head->next = list->tail;
    list->tail->prev = list->head;

    return list;
}


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

static struct test_list_node *test_list_node_alloc(test_case test_case_func)
{
    struct test_list_node *node = calloc(1, sizeof(struct test_list_node));
    if (!node)
    {
        return NULL;
    }

    node->test_case_func = test_case_func;

    return node;
}

static void test_list_node_free(struct test_list_node *const node)
{
    if (!node)
    {
        return;
    }

    free(node);
}

void test_list_foreach(struct test_list *const list, test_list_callback callback, void *user)
{
    struct test_list_node *curr = list->head->next;
    while (curr != list->tail)
    {
        callback(curr->test_name, &curr->result, user);
        curr = curr->next;
    }
}