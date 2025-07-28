#ifndef BACKPROPAGATION_QUEUE_H
#define BACKPROPAGATION_QUEUE_H

#include "autograd/computational_graph/computational_graph.h"
#include <stdlib.h>

struct backpropagation_queue
{
    struct computational_graph_node *data[AUTOGRAD_MAX_NODES];
    size_t front;
    size_t back;
};

static inline cgrad_error backpropagation_queue_init(struct backpropagation_queue *queue);
static inline cgrad_error backpropagation_queue_push(struct backpropagation_queue *queue, struct computational_graph_node *node);
static inline cgrad_error backpropagation_queue_peek(struct backpropagation_queue *queue, struct computational_graph_node **out);
static inline cgrad_error backpropagation_queue_pop(struct backpropagation_queue *queue, struct computational_graph_node **out);
static inline bool backpropagation_queue_is_empty(struct backpropagation_queue *queue);

#endif

static inline cgrad_error backpropagation_queue_init(struct backpropagation_queue *queue)
{
    if (!queue)
    {
        return AUTOGRAD_BACKPROPAGATION_QUEUE_NULL;
    }

    queue->front = 0;
    queue->back = 0;
    memset(queue->data, 0, AUTOGRAD_MAX_NODES * sizeof(struct computational_graph_node *));

    return NO_ERROR;
}

static inline cgrad_error backpropagation_queue_push(struct backpropagation_queue *queue, struct computational_graph_node *node)
{
    if (!queue)
    {
        return AUTOGRAD_BACKPROPAGATION_QUEUE_NULL;
    }
    if (queue->back == AUTOGRAD_MAX_NODES)
    {
        return AUTOGRAD_BACKPROPAGATION_QUEUE_FULL;
    }

    queue->data[queue->back] = node;
    queue->back++;
    return NO_ERROR;
}

static inline cgrad_error backpropagation_queue_peek(struct backpropagation_queue *queue, struct computational_graph_node **out)
{
    if (!queue)
    {
        return AUTOGRAD_BACKPROPAGATION_QUEUE_NULL;
    }
    if (queue->front == queue->back)
    {
        return AUTOGRAD_BACKPROPAGATION_QUEUE_EMPTY;
    }

    (*out) = queue->data[queue->front];
    return NO_ERROR;
}

static inline cgrad_error backpropagation_queue_pop(struct backpropagation_queue *queue, struct computational_graph_node **out)
{
    if (!queue)
    {
        return AUTOGRAD_BACKPROPAGATION_QUEUE_NULL;
    }
    if (queue->front == AUTOGRAD_MAX_NODES)
    {
        (*out) = NULL;
        return AUTOGRAD_BACKPROPAGATION_QUEUE_FULL;
    }
    if (queue->front == queue->back)
    {
        (*out) = NULL;
        return AUTOGRAD_BACKPROPAGATION_QUEUE_EMPTY;
    }

    (*out) = queue->data[queue->front];
    queue->front++;
    return NO_ERROR;
}

static inline bool backpropagation_queue_is_empty(struct backpropagation_queue *queue)
{
    if (!queue)
    {
        return true;
    }

    return queue->front == queue->back;
}