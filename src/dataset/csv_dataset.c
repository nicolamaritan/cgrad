#include "dataset/csv_dataset.h"
#include "config.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

cgrad_error csv_dataset_get_rows(const char *csv_path, size_t *out_rows);
cgrad_error csv_dataset_get_cols(const char *csv_path, size_t *out_cols);
cgrad_error csv_dataset_standard_scale_feature(csv_dataset *dataset, const size_t col, const double mean, const double std_dev);
cgrad_error csv_dataset_standard_compute_mean(csv_dataset *dataset, const size_t col, double *mean);
cgrad_error csv_dataset_standard_compute_std_dev(csv_dataset *dataset, const size_t col, const double mean, double *std_dev);


csv_dataset *csv_dataset_alloc(const char *csv_path)
{
    size_t out_rows = 0;
    size_t out_cols = 0;
    cgrad_error error;

    // Get rows
    error = csv_dataset_get_rows(csv_path, &out_rows);
    if (error != NO_ERROR)
        return NULL;

    // Get cols
    error = csv_dataset_get_cols(csv_path, &out_cols);
    if (error != NO_ERROR)
        return NULL;

    printf("rows %ld\ncols %ld\n", out_rows, out_cols);

    FILE* file = fopen(csv_path, "r");
    if (!file)
    {
        return NULL;
    }

    csv_dataset* dataset = malloc(sizeof(csv_dataset));
    if (!dataset)
    {
        return NULL;
    }
    dataset->csv_data = calloc(sizeof(double), out_cols * out_rows);
    if (!dataset->csv_data)
    {
        return NULL;
    }

    dataset->cols = out_cols;
    dataset->rows = out_rows;
    
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
            // printf("col %d\n", col);
            double field_double = (double)atof(field);
            dataset->csv_data[row * out_cols + col] = field_double;
            field = strtok(NULL, ",");
            col++;
        }

        if (col != dataset->cols)
        {
            printf("error col");
            return NULL;
        }

        row++;
    }
    if (row!= dataset->rows)
    {
        printf("error row");
        return NULL;
    }

    return dataset;
}

cgrad_error csv_dataset_sample_batch_from_permutation(const csv_dataset *const dataset, tensor *const inputs, tensor *const targets, const size_t batch_size, const index_permutation *const permutation)
{
    // TODO error check
    
    size_t cols = dataset->cols;

    for (size_t i = permutation->current; i < permutation->size && i - permutation->current < batch_size; i++)
    {
        size_t batch_idx = i - permutation->current;
        size_t row_idx = permutation->index[i];

        double *csv_row = dataset->csv_data + row_idx * cols;
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
    // TODO check null

    // Skip first column, i.e. label. 
    for (size_t j = 1; j < dataset->cols; j++)
    {
        double mean = 0;
        double std_dev = 0;

        csv_dataset_standard_compute_mean(dataset, j, &mean);
        csv_dataset_standard_compute_std_dev(dataset, j, mean, &std_dev);
        csv_dataset_standard_scale_feature(dataset, j, mean, std_dev);
    }

    return NO_ERROR;
}

cgrad_error csv_dataset_get_rows(const char *csv_path, size_t *out_rows)
{

    FILE* file = fopen(csv_path, "r");
    if (!file)
    {
        return -1;
    }

    // Count \n, without considering header 
    *out_rows = 0;
    int ch = 0;

    // Compute number of rows
    while ((ch = fgetc(file)) != EOF)
    {
        if (ch == '\n')
        {
            (*out_rows)++;
        }
    }

    // Skip first row if at least one row is found
    if (*out_rows >= 1)
    {
        (*out_rows)--;
    }

    fclose(file);
    return NO_ERROR;
}

cgrad_error csv_dataset_get_cols(const char *csv_path, size_t *out_cols)
{
    FILE* file = fopen(csv_path, "r");
    if (!file)
    {
        return -1;
    }

    // Count number of tokens in one line
    (*out_cols) = 0;

    char buffer[DATASET_CSV_MAX_LINE_CHAR_LENGTH];
    fgets(buffer, DATASET_CSV_MAX_LINE_CHAR_LENGTH, file);
    char *field = strtok(buffer, ",");

    while (field)
    {
        (*out_cols)++;
        field = strtok(NULL, ",");
    }

    fclose(file);
    return NO_ERROR;
}

cgrad_error csv_dataset_standard_scale_feature(csv_dataset *dataset, const size_t col, const double mean, const double std_dev)
{
    const double EPS = 10e-8; // Avoid division by zero
    for (size_t i = 0; i < dataset->rows; i++)
    {
        size_t offset = i * dataset->cols + col;
        dataset->csv_data[offset] -= mean;
        dataset->csv_data[offset] /= (std_dev + EPS);
    }

    return NO_ERROR;
}

cgrad_error csv_dataset_standard_compute_mean(csv_dataset *dataset, const size_t col, double *mean)
{
    *mean = 0;

    for (size_t i = 0; i < dataset->rows; i++)
    {
        size_t offset = i * dataset->cols + col;
        (*mean) += dataset->csv_data[offset];
    }

    (*mean) /= dataset->rows;

    return NO_ERROR;
}

cgrad_error csv_dataset_standard_compute_std_dev(csv_dataset *dataset, const size_t col, const double mean, double *std_dev)
{
    *std_dev = 0;

    for (size_t i = 0; i < dataset->rows; i++)
    {
        size_t offset = i * dataset->cols + col;
        double difference = dataset->csv_data[offset] - mean;
        (*std_dev) += difference * difference;
    }

    (*std_dev) /= dataset->rows;
    (*std_dev) = sqrt((*std_dev));

    return NO_ERROR;
}