#ifndef GRAD_TABLE
#define GRAD_TABLE

#include "tensor.h"
#define MAX_GRAD_TABLE 1024

typedef struct
{
    tensor* grad;
} grad_table_entry;

typedef struct {
    grad_table_entry entries[MAX_GRAD_TABLE];
    size_t n_entries;
} grad_table;

static inline void init_grad_table(grad_table* table);
int add_entry(grad_table* const table,  grad_table_entry const entry);
void grad_table_free(grad_table* table);
void grad_table_print(const grad_table* const table);

static inline void init_grad_table(grad_table* table)
{
    table->n_entries = 0;
}

#endif