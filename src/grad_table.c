#include "grad_table.h"
#include <stdlib.h>
#include <stdio.h>

int add_entry(grad_table* const table,  grad_table_entry const entry)
{
    if (table->n_entries >= MAX_GRAD_TABLE)
    {
        return 1;
    }

    table->entries[table->n_entries] = entry;
    table->n_entries++;

    return 0;
}

void grad_table_free(grad_table* table)
{
    // Manually free all gradients
    size_t n_entries = table->n_entries;
    for (size_t i = 0; i < n_entries; i++)
    {
        free(table->entries[i].grad);        
    }

    free(table);
}

void grad_table_print(const grad_table* const table) 
{
    if (table == NULL) {
        fprintf(stderr, "Error: Cannot print NULL grad_table\n");
        return;
    }

    printf("\n=== Grad Table (Entries: %zu) ===\n", table->n_entries);
    
    for (size_t i = 0; i < table->n_entries; ++i) {
        const tensor* grad = table->entries[i].grad;
        printf("Entry %zu:\n", i);
        
        if (grad == NULL) 
        {
            printf("  [NULL tensor]\n");
        }
        else
        {
            printf("  Tensor address: %p\n", (void*)grad);
        }
        
        if (i != table->n_entries - 1) 
        {
            printf("------------------------------\n");
        }
    }
    printf("=== End of Grad Table ===\n\n");
}