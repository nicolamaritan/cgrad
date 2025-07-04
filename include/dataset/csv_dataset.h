#ifndef CSV_DATASET_H
#define CSV_DATASET_H

#include "dataset/indexes_permutation.h"
#include "tensor/tensor.h"
#include "utils/error.h"
#include <stddef.h>

/**
 * @struct csv_dataset
 * @brief Stores a dataset loaded from a CSV file.
 *
 * The first column is assumed to be the label, and the remaining columns are features.
 */
struct csv_dataset 
{
    size_t rows;    /**< Number of data rows (excluding header). */
    size_t cols;    /**< Number of columns (features + label). */
    double *data;   /**< Flattened row-major array of data. */
};

/**
 * @brief Loads a CSV file into a csv_dataset structure.
 *
 * @param csv_path Path to the CSV file.
 * @return Pointer to the allocated csv_dataset, or NULL if allocation failed.
 */
struct csv_dataset *csv_dataset_alloc(const char *csv_path);

/**
 * @brief Samples a batch of data from the dataset using the provided indexes.
 *
 * @param dataset Pointer to the csv_dataset.
 * @param inputs Tensor to store the input features.
 * @param targets Tensor to store the target labels.
 * @param ix_batch Pointer to the indexes_batch specifying which rows to sample.
 * @return NO_ERROR on success, or an error code on failure.
 */
cgrad_error csv_dataset_sample_batch(const struct csv_dataset *const dataset, struct tensor *const inputs, struct tensor *const targets, const struct indexes_batch *const ix_batch);

/**
 * @brief Applies standard scaling (zero mean, unit variance) to the dataset features.
 *
 * The first column (label) is not scaled.
 *
 * @param dataset Pointer to the csv_dataset.
 * @return NO_ERROR on success, or an error code on failure.
 */
cgrad_error csv_dataset_standard_scale(struct csv_dataset *dataset);

/**
 * @brief Checks if the dataset or its data pointer is NULL.
 *
 * @param dataset Pointer to the csv_dataset.
 * @return DATASET_NULL if dataset or data is NULL, NO_ERROR otherwise.
 */
static inline cgrad_error csv_dataset_check_null(const struct csv_dataset *const dataset);

static inline cgrad_error csv_dataset_check_null(const struct csv_dataset *const dataset)
{
    if (!dataset)
    {
        return DATASET_NULL;
    }
    if (!dataset->data)
    {
        return DATASET_NULL;
    }
    return NO_ERROR;
}

#endif