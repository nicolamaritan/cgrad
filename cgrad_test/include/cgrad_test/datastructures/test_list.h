#ifndef TEST_LIST_H 
#define TEST_LIST_H 

#include "cgrad_test/config.h"
#include "cgrad_test/test_result.h"
#include "cgrad_test/test_case.h"
#include <stdlib.h>
#include <stddef.h>

struct test_list_node;

struct test_list_node
{
    struct test_list_node *next;
    struct test_list_node *prev;
    char test_name[TEST_NAME_MAX_SIZE];
    test_case test_case_func;
    struct test_result result;
};

struct test_list
{
    size_t size;
    struct test_list_node *head;
    struct test_list_node *tail;
};

static inline struct test_list *tests_list_alloc();
static inline struct test_list_node *test_list_node_alloc(test_case test_case_func);
static inline void test_list_node_free(struct test_list_node *const node);
static inline void test_list_remove_left(struct test_list *const list);
void test_list_append(struct test_list *const list, test_case test_case_func, const char *test_name);
void test_list_free(struct test_list *const list);

static inline struct test_list *tests_list_alloc()
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

static inline struct test_list_node *test_list_node_alloc(test_case test_case_func)
{
    struct test_list_node *node = calloc(1, sizeof(struct test_list_node));
    if (!node)
    {
        return NULL;
    }

    node->test_case_func = test_case_func;

    return node;
}

static inline void test_list_node_free(struct test_list_node *const node)
{
    if (!node)
    {
        return;
    }

    free(node);
}

#endif
