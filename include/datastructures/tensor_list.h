#ifndef TENSOR_LIST_H
#define TENSOR_LIST_H

#include "tensor/tensor.h"
#include "memory/tensor/tensor_allocator.h"
#include "error.h"
#include <stdlib.h>

struct tensor_list
{
    struct tensor **data;
    size_t capacity;
    size_t size;
};

static inline struct tensor_list *tensor_list_alloc(const size_t capacity);
static inline cgrad_error tensor_list_add(struct tensor_list *const list, struct tensor *const t);
static inline cgrad_error tensor_list_free_all(struct tensor_list *const list, struct tensor_allocator *const tensor_alloc);

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

static inline cgrad_error tensor_list_free_all(struct tensor_list *const list, struct tensor_allocator *const tensor_alloc)
{
    for (size_t i = 0; i < list->size; i++)
    {
        tensor_allocator_free(tensor_alloc, list->data[i]);
    }

    list->size = 0;
    return NO_ERROR;
}

#endif