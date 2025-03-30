#ifndef TENSOR
#define TENSOR

#include <stddef.h>
#include <stdbool.h>

typedef struct computational_graph_node computational_graph_node;
typedef struct tensor tensor;

typedef struct tensor
{
    double *data;
    size_t *shape;
    size_t data_size;
    size_t shape_size;
    computational_graph_node *node;
    tensor *grad;
} tensor;

typedef enum
{
    TENSOR_OK = 0,
    TENSOR_NULL,
    TENSOR_SHAPE_NULL,
    TENSOR_WRONG_SHAPE,
    TENSOR_DATA_NULL,
    TENSOR_INDEX_OUT_OF_BOUNDS,
    TENSOR_SHAPE_MISMATCH,
    TENSOR_DATA_SIZE_MISMATCH
} tensor_error;

tensor *tensor_alloc(size_t *shape, size_t shape_size);
tensor* tensor_no_grad_alloc(size_t *shape, size_t shape_size);
tensor* tensor_no_grad_zero_alloc(size_t *shape, size_t shape_size);
tensor *tensor2d_alloc(size_t rows, size_t cols);
tensor *tensor2d_no_grad_alloc(size_t rows, size_t cols);
tensor *tensor2d_no_grad_zero_alloc(size_t rows, size_t cols);
tensor *tensor2d_alloc_like(tensor *t);
void tensor_free(tensor *t);
void tensor_no_grad_free(tensor *t);

static inline void tensor2d_set_unchecked(tensor *t, size_t row, size_t col, double value);
static inline tensor_error tensor2d_set(tensor *t, size_t row, size_t col, double value);
tensor_error tensor_add_inplace(tensor *A, const tensor *const B);
void tensor_add_inplace_unchecked(tensor *A, const tensor *const B);
tensor *tensor_clone(const tensor *const src);
void tensor2d_copy(const tensor *const src, tensor *const dest);
bool tensor_same_shape(const tensor *const A, const tensor *const B);
tensor_error tensor2d_trans(const tensor *const t, tensor *const out);
void tensor2d_trans_unchecked(const tensor *const t, tensor *const out);

// Differentiable operations
tensor_error tensor_add(const tensor *const A, const tensor *const B, tensor *const out, bool requires_grad);
void tensor_add_unchecked(const tensor *const A, const tensor *const B, tensor *const out, bool requires_grad);

void print_tensor(const tensor *const t);

static inline void tensor2d_set_unchecked(tensor *t, size_t row, size_t col, double value)
{
    t->data[row * t->shape[1] + col] = value;
}

static inline tensor_error tensor2d_set(tensor *t, size_t row, size_t col, double value)
{
    if (t == NULL)
        return TENSOR_NULL;
    if (t->shape == NULL)
        return TENSOR_SHAPE_NULL;
    if (t->data == NULL)
        return TENSOR_DATA_NULL;
    if (row >= t->shape[0] || col >= t->shape[1])
        return TENSOR_INDEX_OUT_OF_BOUNDS;

    t->data[row * t->shape[1] + col] = value;
    return TENSOR_OK;
}

#endif