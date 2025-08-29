#include "cgrad/dataset/csv_dataset.h"
#include "cgrad/config.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Standardizes a single feature column in the dataset.
 *
 * @param dataset Pointer to the csv_dataset.
 * @param col Index of the column to standardize.
 * @param mean Mean of the column.
 * @param std_dev Standard deviation of the column.
 */
static void csv_dataset_standard_scale_feature(struct csv_dataset *dataset, const size_t col, const double mean, const double std_dev);

/**
 * @brief Computes the mean of a feature column.
 *
 * @param dataset Pointer to the csv_dataset.
 * @param col Index of the column.
 * @return Mean value of the column.
 */
static double csv_dataset_standard_compute_mean(struct csv_dataset *dataset, const size_t col);

/**
 * @brief Computes the standard deviation of a feature column.
 *
 * @param dataset Pointer to the csv_dataset.
 * @param col Index of the column.
 * @param mean Mean of the column.
 * @return Standard deviation of the column.
 */
static double csv_dataset_standard_compute_std_dev(struct csv_dataset *dataset, const size_t col, const double mean);

/**
 * @brief Counts the number of rows in the CSV file.
 *
 * @param file Pointer to the open CSV file.
 * @return Number of rows.
 */
static size_t csv_dataset_count_rows(FILE *file);

/**
 * @brief Counts the number of columns in the CSV file.
 *
 * @param file Pointer to the open CSV file.
 * @return Number of columns.
 */
static size_t csv_dataset_count_cols(FILE *file);

/**
 * @brief Fills the dataset's data array from the CSV file.
 *
 * @param dataset Pointer to the csv_dataset.
 * @param file Pointer to the open CSV file.
 * @return NO_ERROR on success, or an error code on failure.
 */
static cgrad_error csv_dataset_fill_data(struct csv_dataset *dataset, FILE *file);

static void copy_features_to_inputs(struct tensor *inputs, double *features, const size_t i, const size_t cols);
static void copy_features_to_inputs_f64(struct tensor *inputs, double *features, const size_t i, const size_t cols);
static void copy_features_to_inputs_f32(struct tensor *inputs, double *features, const size_t i, const size_t cols);
static void copy_label_to_targets(struct tensor *targets, double label, size_t i);
static void copy_label_to_targets_f64(struct tensor *targets, double label, size_t i);
static void copy_label_to_targets_f32(struct tensor *targets, double label, size_t i);

struct csv_dataset *csv_dataset_alloc(const char *csv_path)
{
    size_t rows = 0;
    size_t cols = 0;

    FILE* file = fopen(csv_path, "r");
    if (!file)
    {
        return NULL;
    }

    // Count cols and rows
    cols = csv_dataset_count_cols(file);
    rewind(file);
    rows = csv_dataset_count_rows(file);
    rewind(file);

    // Subtract header line
    if (rows >= 1)
    {
        rows--;
    }

    // CSV dataset allocation
    struct csv_dataset* dataset = malloc(sizeof(struct csv_dataset));
    if (!dataset)
    {
        fclose(file);
        return NULL;
    }
    dataset->data = calloc(sizeof(double), cols * rows);
    if (!dataset->data)
    {
        free(dataset);
        fclose(file);
        return NULL;
    }

    dataset->cols = cols;
    dataset->rows = rows;

    if (csv_dataset_fill_data(dataset, file) != NO_ERROR)
    {
        free(dataset->data);
        free(dataset);
        fclose(file);
        return NULL;
    }

    return dataset;
}

cgrad_error csv_dataset_sample_batch(const struct csv_dataset *const dataset, struct tensor **const inputs, struct tensor **const targets, const struct indexes_batch *const ixs_batch, const cgrad_dtype dtype, struct cgrad_env *const env)
{
    cgrad_error error;
    if ((error = csv_dataset_check_null(dataset) != NO_ERROR))
    {
        return error;
    }
    if (!ixs_batch)
    {
        return INDEXES_BATCH_NULL;
    }
    
    size_t cols = dataset->cols;
    
    size_t inputs_shape[] = {ixs_batch->size, cols - 1};
    (*inputs) = tensor_allocator_alloc(&env->tensor_alloc, inputs_shape, sizeof(inputs_shape) / sizeof(size_t), dtype);
    if (!(*inputs))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    const size_t COLUMN_VECTOR_COLS = 1;
    size_t targets_shape[] = {ixs_batch->size, COLUMN_VECTOR_COLS};
    (*targets) = tensor_allocator_alloc(&env->tensor_alloc, targets_shape, sizeof(targets_shape) / sizeof(size_t), dtype);
    if (!(*targets))
    {
        return TENSOR_ALLOCATION_FAILED;
    }

    for (size_t i = 0; i < ixs_batch->size; i++)
    {
        size_t row_idx = ixs_batch->indexes[i];

        double *csv_row = dataset->data + row_idx * cols;
        double label = csv_row[0];
        double *features = csv_row + 1;

        // Copy features to inputs
        copy_features_to_inputs(*inputs, features, i, cols);
        copy_label_to_targets(*targets, label, i);
    }

    return NO_ERROR;
}

cgrad_error csv_dataset_standard_scale(struct csv_dataset *dataset)
{
    cgrad_error error;
    if ((error = csv_dataset_check_null(dataset) != NO_ERROR))
    {
        return error;
    }

    // Skip first column, i.e. label. 
    for (size_t j = 1; j < dataset->cols; j++)
    {
        double mean = csv_dataset_standard_compute_mean(dataset, j);
        double std_dev = csv_dataset_standard_compute_std_dev(dataset, j, mean);
        csv_dataset_standard_scale_feature(dataset, j, mean, std_dev);
    }

    return NO_ERROR;
}

static void csv_dataset_standard_scale_feature(struct csv_dataset *dataset, const size_t col, const double mean, const double std_dev)
{
    const double EPS = 10e-8; // Avoid division by zero
    for (size_t i = 0; i < dataset->rows; i++)
    {
        size_t offset = i * dataset->cols + col;
        dataset->data[offset] -= mean;
        dataset->data[offset] /= (std_dev + EPS);
    }
}

static double csv_dataset_standard_compute_mean(struct csv_dataset *dataset, const size_t col)
{
    double mean = 0;

    for (size_t i = 0; i < dataset->rows; i++)
    {
        size_t offset = i * dataset->cols + col;
        mean += dataset->data[offset];
    }

    mean /= dataset->rows;
    return mean;
}

static double csv_dataset_standard_compute_std_dev(struct csv_dataset *dataset, const size_t col, const double mean)
{
    double std_dev = 0;
    for (size_t i = 0; i < dataset->rows; i++)
    {
        size_t offset = i * dataset->cols + col;
        double difference = dataset->data[offset] - mean;
        std_dev += difference * difference;
    }

    std_dev /= dataset->rows;
    std_dev = sqrt(std_dev);

    return std_dev;
}

static size_t csv_dataset_count_rows(FILE *file)
{
    size_t rows = 0;
    int ch;
    while ((ch = fgetc(file)) != EOF)
    {
        if (ch == '\n')
        {
            rows++;
        }
    }
    return rows;
}

static size_t csv_dataset_count_cols(FILE *file)
{
    size_t cols = 0;
    char buffer[DATASET_CSV_MAX_LINE_CHAR_LENGTH];
    fgets(buffer, DATASET_CSV_MAX_LINE_CHAR_LENGTH, file);
    char *field = strtok(buffer, ",");

    while (field)
    {
        cols++;
        field = strtok(NULL, ",");
    }

    return cols;
}

static cgrad_error csv_dataset_fill_data(struct csv_dataset *dataset, FILE *file)
{
    char buffer[DATASET_CSV_MAX_LINE_CHAR_LENGTH];

    // Skip first row
    fgets(buffer, DATASET_CSV_MAX_LINE_CHAR_LENGTH, file);

    size_t row = 0;
    while (fgets(buffer, DATASET_CSV_MAX_LINE_CHAR_LENGTH, file))
    {
        size_t col = 0;
        char *field = strtok(buffer, ",");
        while (field)
        {
            double field_double = (double)atof(field);
            dataset->data[row * dataset->cols + col] = field_double;
            field = strtok(NULL, ",");
            col++;
        }

        if (col != dataset->cols)
        {
            return CSV_DATASET_FORMAT_ERROR;
        }

        row++;
    }
    if (row!= dataset->rows)
    {
        return CSV_DATASET_FORMAT_ERROR;
    }

    return NO_ERROR;
}

static void copy_features_to_inputs(struct tensor *inputs, double *features, const size_t i, const size_t cols)
{
    switch (inputs->dtype)
    {
        case DTYPE_FLOAT64:
            copy_features_to_inputs_f64(inputs, features, i, cols);
            break;
        case DTYPE_FLOAT32:
            copy_features_to_inputs_f32(inputs, features, i, cols);
            break;
        default:
            break;
    }
}

static void copy_features_to_inputs_f64(struct tensor *inputs, double *features, const size_t i, const size_t cols)
{
    double *inputs_data = (double *)inputs->data; // Cast is needed for correct pointer arithmetic below
    memcpy(inputs_data + i * (cols - 1), features, (cols - 1) * sizeof(double));
}

static void copy_features_to_inputs_f32(struct tensor *inputs, double *features, const size_t i, const size_t cols)
{
    float *inputs_data = (float *)inputs->data; // Cast is needed for correct pointer arithmetic below
    for (size_t j = 0; j < cols - 1; j++)
    {
        inputs_data[i * (cols - 1) + j] = features[j];
    }
}

static void copy_label_to_targets(struct tensor *targets, double label, const size_t i)
{
    switch (targets->dtype)
    {
        case DTYPE_FLOAT64:
            copy_label_to_targets_f64(targets, label, i);
            break;
        case DTYPE_FLOAT32:
            copy_label_to_targets_f32(targets, label, i);
            break;
        default:
            break;
    }
}

static void copy_label_to_targets_f64(struct tensor *targets, double label, const size_t i)
{
    double *targets_data = (double *)targets->data;
    targets_data[i] = label;
}

static void copy_label_to_targets_f32(struct tensor *targets, double label, const size_t i)
{
    float *targets_data = (float *)targets->data;
    targets_data[i] = label;
}