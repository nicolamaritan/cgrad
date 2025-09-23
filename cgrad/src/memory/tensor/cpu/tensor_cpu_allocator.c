#include "cgrad/memory/tensor/cpu/tensor_cpu_allocator.h"
#include <string.h>

/**
 * @brief Allocates a CPU tensor and its gradient if applicable.
 *
 * This function allocates a tensor from the given CPU pool. For real-valued
 * tensors (DTYPE_FLOAT32 or DTYPE_FLOAT64), it also allocates a gradient tensor.
 * If any allocation fails, previously allocated memory is freed and NULL is returned.
 *
 * @param[in] pool Pointer to the CPU tensor pool.
 * @param[in] shape Array containing the size of each dimension of the tensor.
 * @param[in] shape_size Number of dimensions of the tensor.
 * @param[in] dtype Data type of the tensor (see @ref cgrad_dtype).
 *
 * @return Pointer to the allocated tensor, or NULL if allocation failed.
 *
 * @note The returned tensor should be freed with tensor_cpu_free().
 * @note Gradient tensors are only allocated for real-valued types.
 */
static struct tensor *tensor_cpu_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Allocates a CPU tensor without gradient.
 *
 * This function allocates memory for the tensor's data from the CPU pool,
 * computes its stride, and initializes all tensor fields. No gradient tensor
 * is allocated.
 *
 * @param[in] pool Pointer to the CPU tensor pool.
 * @param[in] shape Array containing the size of each dimension of the tensor.
 * @param[in] shape_size Number of dimensions of the tensor.
 * @param[in] dtype Data type of the tensor (see @ref cgrad_dtype).
 *
 * @return Pointer to the allocated tensor, or NULL if allocation fails.
 *
 * @note This function does not allocate a gradient tensor.
 */
static struct tensor *tensor_cpu_no_grad_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Allocates a CPU tensor without gradient and initializes its data to zero.
 *
 * This function allocates memory for the tensor's data from the CPU pool, sets
 * all elements to zero, computes its stride, and initializes all tensor fields.
 * No gradient tensor is allocated.
 *
 * @param[in] pool Pointer to the CPU tensor pool.
 * @param[in] shape Array containing the size of each dimension of the tensor.
 * @param[in] shape_size Number of dimensions of the tensor.
 * @param[in] dtype Data type of the tensor (see @ref cgrad_dtype).
 *
 * @return Pointer to the allocated tensor, or NULL if allocation fails.
 *
 * @note This function does not allocate a gradient tensor.
 */
static struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Allocates a CPU tensor and copies data from an existing array.
 *
 * This function allocates a tensor from the CPU pool and copies the contents
 * from the provided array into the tensor's data buffer.
 *
 * @param[in] pool Pointer to the CPU tensor pool.
 * @param[in] data Pointer to the source data array.
 * @param[in] shape Array containing the size of each dimension of the tensor.
 * @param[in] shape_size Number of dimensions of the tensor.
 * @param[in] dtype Data type of the tensor (see @ref cgrad_dtype).
 *
 * @return Pointer to the allocated tensor, or NULL if allocation fails.
 *
 * @note The size of the data array must match the total number of elements
 *       defined by shape and shape_size.
 */
static struct tensor *tensor_cpu_from_array_alloc(void *pool, const void *data, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype);

/**
 * @brief Frees a CPU tensor and its associated data.
 *
 * This function releases the memory of the tensor's data and gradient (if any),
 * and returns the tensor structure back to the CPU pool. The node pointer is
 * not freed by this function and should be managed separately if used.
 *
 * @param[in] pool Pointer to the CPU tensor pool.
 * @param[in,out] t Pointer to the tensor to free.
 *
 * @note If t is NULL, the function does nothing.
 * @note The tensor's node pointer is not freed; it is the caller's responsibility
 *       to handle it if necessary.
 * @note Use this function to properly free tensors allocated with
 *       tensor_cpu_alloc() or tensor_cpu_no_grad_alloc().
 */
static void tensor_cpu_free(void *pool, struct tensor *t);

/**
 * @brief Frees a CPU tensor without gradient.
 *
 * This function releases the memory of the tensor's data and returns the
 * tensor structure back to the CPU pool. No gradient tensor is involved.
 *
 * @param[in] pool Pointer to the CPU tensor pool.
 * @param[in,out] t Pointer to the tensor to free.
 *
 * @note If t is NULL, the function does nothing.
 * @note Use this function to properly free tensors allocated with
 *       tensor_cpu_no_grad_alloc().
 *
 */
static void tensor_cpu_no_grad_free(void *pool, struct tensor *t);

/**
 * @brief Clones a CPU tensor.
 *
 * This function allocates a new tensor from the CPU pool with the same shape
 * and data type as the source tensor, and copies its data into the new tensor.
 *
 * @param[in] pool Pointer to the CPU tensor pool.
 * @param[in] src Pointer to the source tensor to clone.
 *
 * @return Pointer to the newly allocated tensor, or NULL if allocation fails
 *         or if the source tensor is NULL.
 *
 * @note This function currently copies only the first two dimensions using
 *       sizeof(double); adjust if using other data types or higher dimensions.
 */
static struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src);

/**
 * @brief Computes the stride array for a tensor given its shape.
 *
 * The stride array indicates the number of elements to skip in memory
 * to move along each dimension. This is used for indexing multi-dimensional
 * tensors stored in a flat array.
 *
 * @param[in]  shape      Array containing the size of each dimension of the tensor.
 * @param[out] stride     Array where the computed stride for each dimension will be stored.
 * @param[in]  shape_size Number of dimensions of the tensor.
 *
 * @note The stride array must be pre-allocated with at least shape_size elements.
 * @note This function assumes row-major ordering
 */
static void compute_stride(size_t *const shape, size_t *const stride, size_t const shape_size);

cgrad_error tensor_cpu_allocator_init(struct tensor_allocator *const tensor_alloc)
{
    if (!tensor_alloc)
    {
        return TENSOR_ALLOCATOR_NULL;
    }

    struct tensor_cpu_pool *tensor_pool = calloc(1, sizeof(struct tensor_cpu_pool));
    if (!tensor_pool)
    {
        return TENSOR_POOL_ALLOCATION_FAILED;
    }

    cgrad_error err = tensor_cpu_pool_init(tensor_pool);
    if (err != NO_ERROR)
    {
        return err;
    }

    tensor_alloc->alloc = tensor_cpu_alloc,
    tensor_alloc->no_grad_alloc = tensor_cpu_no_grad_alloc,
    tensor_alloc->no_grad_zero_alloc = tensor_cpu_no_grad_zero_alloc,
    tensor_alloc->from_array_alloc = tensor_cpu_from_array_alloc;
    tensor_alloc->free = tensor_cpu_free,
    tensor_alloc->no_grad_free = tensor_cpu_no_grad_free,
    tensor_alloc->clone = tensor_cpu_clone,
    tensor_alloc->pool = tensor_pool;

    return NO_ERROR;
}

void tensor_cpu_allocator_cleanup(struct tensor_allocator *const tensor_alloc)
{
    if (!tensor_alloc)
    {
        return;
    }

    tensor_cpu_pool_cleanup(tensor_alloc->pool);
    free(tensor_alloc->pool);
}

static struct tensor *tensor_cpu_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_cpu_no_grad_alloc(cpu_pool, shape, shape_size, dtype);
    if (!t)
    {
        return NULL;
    }

    // Allocate gradient only for real value tensors
    if (dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64)
    {
        t->grad = tensor_cpu_no_grad_zero_alloc(cpu_pool, shape, shape_size, dtype);
        if (!t->grad)
        {
            tensor_cpu_free(cpu_pool, t);
            return NULL;
        }
    }
    else
    {
        t->grad = NULL;
    }
    return t;
}

static struct tensor *tensor_cpu_no_grad_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_cpu_pool_tensor_alloc(cpu_pool);
    if (!t)
    {
        return NULL;
    }

    void *data = tensor_cpu_pool_data_zero_alloc(cpu_pool, data_size * dtype_sizeof(dtype));
    if (!data)
    {
        tensor_cpu_pool_tensor_free(cpu_pool, t);
        return NULL;
    }

    // Init _shape
    memcpy(t->shape, shape, shape_size * sizeof(size_t));

    compute_stride(t->shape, t->stride, shape_size);

    t->data = data;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;
    t->dtype = dtype;

    return t;
}

static struct tensor *tensor_cpu_no_grad_zero_alloc(void *pool, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    // Compute data_size, needed for data allocation
    size_t data_size = 1;
    for (size_t i = 0; i < shape_size; i++)
    {
        data_size *= shape[i];
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    struct tensor *t = tensor_cpu_pool_tensor_alloc(cpu_pool);
    if (!t)
    {
        return NULL;
    }

    void *data = tensor_cpu_pool_data_zero_alloc(cpu_pool, data_size * dtype_sizeof(dtype));
    if (!data)
    {
        tensor_cpu_pool_tensor_free(cpu_pool, t);
        return NULL;
    }

    // Init _shape
    memcpy(t->shape, shape, shape_size * sizeof(size_t));

    compute_stride(t->shape, t->stride, shape_size);

    t->data = data;
    t->node = NULL;
    t->data_size = data_size;
    t->shape_size = shape_size;
    t->grad = NULL;
    t->dtype = dtype;

    return t;
}

static struct tensor *tensor_cpu_from_array_alloc(void *pool, const void *data, const size_t *const shape, const size_t shape_size, const cgrad_dtype dtype)
{
    struct tensor *t = tensor_cpu_alloc(pool, shape, shape_size, dtype);
    if (!t)
    {
        return NULL;
    }

    memcpy(t->data, data, t->data_size * dtype_sizeof(dtype));

    return t;
}

static void tensor_cpu_free(void *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    tensor_cpu_pool_data_free(cpu_pool, t->data);
    t->data = NULL;

    if (t->grad)
    {
        tensor_cpu_no_grad_free(cpu_pool, t->grad);
        t->grad = NULL;
    }

    if (t->node)
    {
        t->node = NULL; // The node will be freed separately
    }

    tensor_cpu_pool_tensor_free(cpu_pool, t);
}

static void tensor_cpu_no_grad_free(void *pool, struct tensor *t)
{
    if (!t)
    {
        return;
    }

    struct tensor_cpu_pool *cpu_pool = (struct tensor_cpu_pool *)pool;
    tensor_cpu_pool_data_free(cpu_pool, t->data);
    t->data = NULL;

    tensor_cpu_pool_tensor_free(cpu_pool, t);
}

static struct tensor *tensor_cpu_clone(void *pool, const struct tensor *const src)
{
    if (!src)
    {
        return NULL;
    }

    struct tensor *new_tensor = tensor_cpu_alloc(pool, src->shape, src->shape_size, src->dtype);
    if (!new_tensor)
    {
        return NULL;
    }

    memcpy(new_tensor->data, src->data, src->shape[0] * src->shape[1] * sizeof(double));
    return new_tensor;
}

static void compute_stride(size_t *const shape, size_t *const stride, size_t const shape_size)
{
    stride[shape_size - 1] = 1;
    // Use int for allowing i = 0
    for (int i = shape_size - 2; i >= 0; i--)
    {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}