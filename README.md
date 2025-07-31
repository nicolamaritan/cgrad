# CGrad
Deep Learning library for the C programming language.

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

## Features
- Tensor library
- Dynamic computational graph construction
- Automatic tensor differentiation via backpropagation
- Modular operation system with custom backward functions
- Custom memory management for fast allocations
- SIMD and BLAS-accelerated computations for performance

## Notes
- Currently supports CPU only - no GPU acceleration yet

## Examples

Some examples are provided inside the `examples` directory to demonstrate the use of the library.

### Regression example
The `mlp_regression.c` example fits a 2-layers MLP with ReLU activation on the `tanh` of a random linear combination of the inputs.

1. Build the project using `make`.
2. Execute the example executable:

```bash
./examples/mlp_regression.out
```

### MNIST classification example
The `mlp_mnist_classification.c` example fits a 2-layers MLP with ReLU activation on a flattened version of the MNIST dataset.

1. Build the project using `make`.
2. Execute the example executable:

```bash
./examples/mlp_mnist_classification.out <mnist_train_dataset_path>
```
