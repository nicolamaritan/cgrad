#include "cgrad/tensor/tensor_equality.h"
#include <math.h>

static bool tensor_no_grad_same_data_f32(const struct tensor *const t1, const struct tensor *const t2);
static bool tensor_no_grad_same_data_f64(const struct tensor *const t1, const struct tensor *const t2);

bool tensor_no_grad_equal(const struct tensor *const t1, const struct tensor *const t2)
{
    if ((t1 && !t2) || (!t1 && t2))
    {
        return false;
    }
    if (!t1 && !t2)
    {
        return true;
    }
    if (t1->dtype != t2->dtype)
    {
        return false;
    }

    if (!tensor_same_shape(t1, t2))
    {
        return false;
    }

    return tensor_no_grad_same_data(t1, t2);
}

bool tensor_equal(const struct tensor *const t1, const struct tensor *const t2)
{
    return tensor_no_grad_equal(t1, t2) && tensor_no_grad_equal(t1->grad, t2->grad);
}

bool tensor_same_shape(const struct tensor *const t1, const struct tensor *const t2)
{
    if ((t1 && !t2) || (!t1 && t2))
    {
        return false;
    }
    if (!t1 && !t2)
    {
        return true;
    }
    if (t1->shape_size != t2->shape_size)
    {
        return false;
    }

    size_t shape_size = t1->shape_size;
    for (size_t i = 0; i < shape_size; i++)
    {
        if (t1->shape[i] != t2->shape[i])
        {
            return false;
        }
    }
    return true;
}

inline bool tensor_no_grad_same_data(const struct tensor *const t1, const struct tensor *const t2)
{
    if (!t1->data && !t2->data)
    {
        return true;
    }
    if ((!t1->data && t2->data) || (t1->data && !t2->data))
    {
        return false;
    }
    if (t1->data_size != t2->data_size)
    {
        return false;
    }

    switch (t1->dtype)
    {
    case DTYPE_FLOAT32:
        return tensor_no_grad_same_data_f32(t1, t2);
    case DTYPE_FLOAT64:
        return tensor_no_grad_same_data_f64(t1, t2);
    default:
        return false;
    }
}

static bool tensor_no_grad_same_data_f32(const struct tensor *const t1, const struct tensor *const t2)
{
    const float EPS = 1e-6;
    float *t1_data = t1->data;
    float *t2_data = t2->data;

    for (size_t i = 0; i < t1->data_size; i++)
    {
        if (fabs(t1_data[i] - t2_data[i]) > EPS)
        {
            return false;
        }
    }

    return true;
}

static bool tensor_no_grad_same_data_f64(const struct tensor *const t1, const struct tensor *const t2)
{
    const double EPS = 1e-6;
    double *t1_data = t1->data;
    double *t2_data = t2->data;

    for (size_t i = 0; i < t1->data_size; i++)
    {
        if (fabs(t1_data[i] - t2_data[i]) > EPS)
        {
            return false;
        }
    }

    return true;
}