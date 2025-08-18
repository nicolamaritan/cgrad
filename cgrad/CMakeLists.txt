set(CMAKE_C_FLAGS_RELEASE "-Wall -Iinclude -mavx2 -DENABLE_SIMD_AVX2 -DNDEBUG -O3")
set(CMAKE_C_FLAGS_DEBUG "-Wall -Iinclude -mavx2 -DENABLE_SIMD_AVX2 -g")

add_library(cgrad STATIC

    # Autograd sources
    src/autograd/backpropagation/backpropagation.c
    src/autograd/computational_graph/computational_graph.c
    src/autograd/computational_graph/computational_graph_link.c

    # Dataset sources
    src/dataset/csv_dataset.c
    src/dataset/indexes_batch.c
    src/dataset/indexes_permutation.c

    # Layers sources
    src/layers/conv2d/conv2d.c
    src/layers/linear/linear.c
    src/layers/relu.c

    # Losses sources
    src/losses/cross_entropy.c
    src/losses/mse.c

    # Memory sources
    src/memory/computational_graph/computational_graph_cpu_allocator.c
    src/memory/computational_graph/computational_graph_cpu_pool.c
    src/memory/tensor/cpu/tensor_cpu_allocator.c
    src/memory/tensor/cpu/tensor_cpu_pool.c

    # Model sources
    src/model/model_params.c

    # Optimizers sources
    src/optimizers/sgd.c

    # Tensor sources
    src/tensor/tensor2d_add_row_vector.c
    src/tensor/tensor2d_mult.c
    src/tensor/tensor2d_mult_lhs_trans.c
    src/tensor/tensor2d_mult_rhs_trans.c
    src/tensor/tensor2d_trans.c
    src/tensor/tensor_add.c
    src/tensor/tensor_add_inplace.c
    src/tensor/tensor_axpy.c
    src/tensor/tensor_copy.c
    src/tensor/tensor_get.c
    src/tensor/tensor_helpers.c
    src/tensor/tensor_im2row.c
    src/tensor/tensor_norm.c
    src/tensor/tensor_reshape.c
    src/tensor/tensor_scalar_mult_tensor_add.c
    src/tensor/tensor_set.c
    src/tensor/tensor_sum.c
    src/tensor/tensor_trans.c
)

target_compile_options(cgrad PRIVATE
    $<$<CONFIG:Release>:-Wall -mavx2 -DENABLE_SIMD_AVX2 -DNDEBUG -O3>
    $<$<CONFIG:Debug>:-Wall -mavx2 -DENABLE_SIMD_AVX2 -g>
)

target_include_directories(cgrad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cgrad PUBLIC
    m
    blas
)