# C-Grad

Lightweight C neural network library.

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

An example MLP is provided to demonstrate the use of the library. To run the MLP example:

1. Build the project using `make` (or `make debug` for debugging).
2. Execute the MLP example executable:

```bash
./examples/mlp_example.out
```