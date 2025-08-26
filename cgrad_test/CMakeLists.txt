add_library(cgrad_test STATIC
    src/datastructures/test_list.c
    src/run_tests.c
    src/tensor2d_mult_tests.c
)

target_include_directories(cgrad_test PUBLIC
    ${CMAKE_SOURCE_DIR}/cgrad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cgrad_test PUBLIC
    cgrad
)

add_executable(main main.c)
target_link_libraries(main PRIVATE
    cgrad
    cgrad_test
)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/cgrad_test/include)