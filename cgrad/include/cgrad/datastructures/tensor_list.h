#ifndef TENSOR_LIST_H
#define TENSOR_LIST_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/memory/tensor/tensor_allocator.h"
#include "cgrad/error.h"
#include <stdlib.h>

struct tensor_list
{
    struct tensor **data;
    size_t capacity;
    size_t size;
};

static inline struct tensor_list *tensor_list_alloc(const size_t capacity);
static inline cgrad_error tensor_list_add(struct tensor_list *const list, struct tensor *const t);
static inline void tensor_list_free(struct tensor_list *const list);

static inline struct tensor_list *tensor_list_alloc(const size_t capacity)
{
    struct tensor_list *list = (struct tensor_list *)malloc(sizeof(struct tensor_list));
    if (!list)
    {
        return NULL;
    }

    list->data = (struct tensor **)calloc(capacity, sizeof(struct tensor *));
    if (!list->data)
    {
        free(list);
        return NULL;
    }

    list->size = 0;
    list->capacity = capacity;

    return list;
}

static inline cgrad_error tensor_list_add(struct tensor_list *const list, struct tensor *const t)
{
    if (!list)
    {
        return TENSOR_LIST_NULL;
    }
    if (!t)
    {
        return TENSOR_NULL;
    }
    if (list->size == list->capacity)
    {
        return TENSOR_LIST_FULL;
    }

    list->data[list->size++] = t;
    return NO_ERROR;
}

static inline void tensor_list_free(struct tensor_list *const list)
{
    if (!list)
    {
        return;
    }

    free(list->data);
    list->data = NULL;
    free(list);
}

#endif