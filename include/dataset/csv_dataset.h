#ifndef CSV_DATASET
#define CSV_DATASET

#include "dataset/index_permutation.h"
#include "tensor/tensor.h"
#include "utils/error.h"
#include <stddef.h>

typedef struct csv_dataset 
{
    size_t rows;
    size_t cols;
    double *csv_data;
} csv_dataset;

csv_dataset *csv_dataset_alloc(const char *csv_path);
cgrad_error csv_dataset_sample_batch_from_permutation(const csv_dataset *const dataset, tensor *const inputs, tensor *const targets, const size_t batch_size, const index_permutation *const permutation);
cgrad_error csv_dataset_standard_scale(csv_dataset *dataset);


#endif