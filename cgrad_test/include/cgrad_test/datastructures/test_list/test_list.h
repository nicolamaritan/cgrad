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

typedef void (*test_list_callback)(const char *test_name, struct test_result *result, void *user);

struct test_list *tests_list_alloc();
void test_list_append(struct test_list *const list, test_case test_case_func, const char *test_name);
void test_list_free(struct test_list *const list);
void test_list_foreach(struct test_list *const list, test_list_callback callback, void *user);

#endif
