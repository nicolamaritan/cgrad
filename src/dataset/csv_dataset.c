#include "dataset/csv_dataset.h"
#include "config.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

cgrad_error csv_dataset_get_rows(const char *csv_path, size_t *out_rows);
cgrad_error csv_dataset_get_cols(const char *csv_path, size_t *out_cols);

csv_dataset *csv_dataset_alloc(const char *csv_path)
{
    size_t out_rows = 0;
    size_t out_cols = 0;
    cgrad_error error;
    error = csv_dataset_get_rows(csv_path, &out_rows);
    error = csv_dataset_get_cols(csv_path, &out_cols);

    printf("rows %ld\ncols %ld\n", out_rows, out_cols);

    FILE* file = fopen(csv_path, "r");
    if (!file)
    {
        return NULL;
    }

    csv_dataset* dataset = malloc(sizeof(dataset));
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
    
    size_t current_index = 0;
    char buffer[DATASET_CSV_MAX_LINE_CHAR_LENGTH];

    // Skip first row
    fgets(buffer, DATASET_CSV_MAX_LINE_CHAR_LENGTH, file);

    size_t row = 0;
    while (fgets(buffer, DATASET_CSV_MAX_LINE_CHAR_LENGTH, file))
    {
        printf("row %d\n", row);
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

cgrad_error csv_dataset_sample_batch_from_permutation(const csv_dataset *const dataset, tensor *const t, const size_t batch_size, const index_permutation *const permutation)
{
    // TODO error check
    // Since t is a 2D tensor, we can just memcpy the data
    
    double *csv_data_to_copy = dataset->csv_data + permutation->current;
    memcpy(t->data, csv_data_to_copy, batch_size);
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
    if (out_rows >= 1)
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