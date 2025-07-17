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
        return 1;
    }

    queue->front = 0;
    queue->back = 0;
    memset(queue->data, 0, AUTOGRAD_MAX_NODES * sizeof(struct computational_graph_node *));

    return NO_ERROR;
}

static inline cgrad_error backpropagation_queue_push(struct backpropagation_queue *queue, struct computational_graph_node *node)
{
    if (queue->back == AUTOGRAD_MAX_NODES)
    {
        return 1;
    }

    queue->data[queue->back] = node;
    queue->back++;
    return NO_ERROR;
}

static inline cgrad_error backpropagation_queue_peek(struct backpropagation_queue *queue, struct computational_graph_node **out)
{
    (*out) = queue->data[queue->front];
    return NO_ERROR;
}

static inline cgrad_error backpropagation_queue_pop(struct backpropagation_queue *queue, struct computational_graph_node **out)
{
    if (queue->front == AUTOGRAD_MAX_NODES)
    {
        return 1;
    }
    if (queue->front == queue->back)
    {
        return 1;
    }

    (*out) = queue->data[queue->front];
    queue->front++;
    return NO_ERROR;
}

static inline bool backpropagation_queue_is_empty(struct backpropagation_queue *queue)
{
    if (!queue)
    {
        return 1;
    }

    return queue->front == queue->back;
}