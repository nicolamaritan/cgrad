#ifndef COMPUTATIONAL_GRAPH
#define COMPUTATIONAL_GRAPH

#include "backpropagation_function.h"

#define MAX_PARENTS 32
#define MAX_CHILDREN 32
#define MAX_TARGETS 1024

typedef struct computational_graph_node computational_graph_node;

typedef struct computational_graph_node
{
    tensor* t;
    size_t grad_table_index;
    size_t n_parents;
    size_t n_children;
    computational_graph_node* parents[MAX_PARENTS];
    size_t parents_operands[MAX_PARENTS];
    computational_graph_node* children[MAX_CHILDREN];
    backpropagation_function_data* data;
    backpropagation_function function;
} computational_graph_node;

typedef struct 
{
    computational_graph_node* targets[MAX_TARGETS];
    size_t size;
} target_computational_graph_nodes;

typedef struct 
{
    computational_graph_node* root;
    size_t size;
} computational_graph;

computational_graph_node* computational_graph_node_alloc();
computational_graph_node* computational_graph_node_tensor_alloc(tensor* t);
int add_child(computational_graph_node* const node,  computational_graph_node* const child);
int add_parent(computational_graph_node* const node,  computational_graph_node* const parent, const size_t operand);
int add_target(target_computational_graph_nodes* const targets, computational_graph_node* const node);
void print_computational_graph_node(const computational_graph_node* node);


#endif