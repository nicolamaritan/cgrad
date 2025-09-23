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

/**
 * @brief Allocates and initializes a tensor list with a specified capacity.
 *
 * This function creates a `tensor_list` structure, allocates memory for
 * an array of tensor pointers, and sets its size and capacity. The
 * `data` array is zero-initialized.
 *
 * @param capacity The maximum number of tensors the list can hold.
 *
 * @return Pointer to the newly allocated `tensor_list` on success, or
 *         NULL if memory allocation fails.
 *
 * @note The `size` field is initialized to 0, indicating the list is empty.
 *
 */
static inline struct tensor_list *tensor_list_alloc(const size_t capacity);

/**
 * @brief Adds a tensor to a tensor list.
 *
 * This function appends the given tensor `t` to the `tensor_list` if
 * there is available capacity.
 *
 * @param list Pointer to the `tensor_list` to add the tensor to. Must not be NULL.
 * @param t Pointer to the `tensor` to add. Must not be NULL.
 *
 * @return 
 * - `NO_ERROR` if the tensor was successfully added.
 * - `TENSOR_LIST_NULL` if `list` is NULL.
 * - `TENSOR_NULL` if `t` is NULL.
 * - `TENSOR_LIST_FULL` if the list has reached its capacity.
 */
static inline cgrad_error tensor_list_add(struct tensor_list *const list, struct tensor *const t);

/**
 * @brief Frees a tensor list and its internal data array.
 *
 * This function releases the memory allocated for the `tensor_list`
 * structure and its internal `data` array. It does not free the tensors
 * contained in the list; those must be freed separately.
 *
 * @param list Pointer to the `tensor_list` to free. If NULL, the function does nothing.
 */
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