#include "dataset/csv_dataset.h"
#include "config.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void csv_dataset_standard_scale_feature(csv_dataset *dataset, const size_t col, const double mean, const double std_dev);
static double csv_dataset_standard_compute_mean(csv_dataset *dataset, const size_t col);
static double csv_dataset_standard_compute_std_dev(csv_dataset *dataset, const size_t col, const double mean);
static size_t csv_dataset_count_rows(FILE *file);
static size_t csv_dataset_count_cols(FILE *file);
static cgrad_error csv_dataset_fill_data(csv_dataset *dataset, FILE *file);

csv_dataset *csv_dataset_alloc(const char *csv_path)
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
    csv_dataset* dataset = malloc(sizeof(csv_dataset));
    if (!dataset)
    {
        return NULL;
    }
    dataset->data = calloc(sizeof(double), cols * rows);
    if (!dataset->data)
    {
        return NULL;
    }

    dataset->cols = cols;
    dataset->rows = rows;

    if (csv_dataset_fill_data(dataset, file) != NO_ERROR)
    {
        return NULL;
    }

    return dataset;
}

cgrad_error csv_dataset_sample_batch_from_permutation(const csv_dataset *const dataset, tensor *const inputs, tensor *const targets, const size_t batch_size, const index_permutation *const permutation)
{
    cgrad_error error;
    if ((error = tensor_check_null(inputs)) != NO_ERROR)
    {
        return error;
    }
    if ((error = tensor_check_null(targets)) != NO_ERROR)
    {
        return error;
    }
    if (!(error = csv_dataset_check_null(dataset) != NO_ERROR))
    {
        return error;
    }
    if (!permutation)
    {
        return PERMUTATION_NULL;
    }
    
    size_t cols = dataset->cols;

    for (size_t i = permutation->current; i < permutation->size && i - permutation->current < batch_size; i++)
    {
        size_t batch_idx = i - permutation->current;
        size_t row_idx = permutation->index[i];

        double *csv_row = dataset->data + row_idx * cols;
        double label = csv_row[0];
        double *features = csv_row + 1;

        // Copy features to inputs
        memcpy(inputs->data + batch_idx * (cols - 1), features, (cols - 1) * sizeof(double));

        targets->data[batch_idx] = label;
    }

    return NO_ERROR;
}

cgrad_error csv_dataset_standard_scale(csv_dataset *dataset)
{
    cgrad_error error;
    if (!(error = csv_dataset_check_null(dataset) != NO_ERROR))
    {
        return error;
    }

    // Skip first column, i.e. label. 
    for (size_t j = 1; j < dataset->cols; j++)
    {
        double mean = 0;
        double std_dev = 0;

        csv_dataset_standard_compute_mean(dataset, j);
        csv_dataset_standard_compute_std_dev(dataset, j, mean);
        csv_dataset_standard_scale_feature(dataset, j, mean, std_dev);
    }

    return NO_ERROR;
}

static void csv_dataset_standard_scale_feature(csv_dataset *dataset, const size_t col, const double mean, const double std_dev)
{
    const double EPS = 10e-8; // Avoid division by zero
    for (size_t i = 0; i < dataset->rows; i++)
    {
        size_t offset = i * dataset->cols + col;
        dataset->data[offset] -= mean;
        dataset->data[offset] /= (std_dev + EPS);
    }
}

static double csv_dataset_standard_compute_mean(csv_dataset *dataset, const size_t col)
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

static double csv_dataset_standard_compute_std_dev(csv_dataset *dataset, const size_t col, const double mean)
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

static cgrad_error csv_dataset_fill_data(csv_dataset *dataset, FILE *file)
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