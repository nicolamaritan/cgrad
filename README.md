# C-Grad
Neural network library written in C.

## Dependencies
- libblas-dev

## Build 

A Makefile is provided in the project root. To build the project, run:

```bash
make
```

This command compiles the source files in `src/` and builds all examples in the `examples/` directory.

For debug builds, run:

```bash
make debug
```

## Example 

Some examples are provided inside the `examples` directory to demonstrate the use of the library. For instance, to train an MLP on the MNIST dataset:

1. Build the project using `make` (or `make debug` for debugging).
2. Execute the MLP example executable:

```bash
./examples/mlp_mnist_classification_example.out
```