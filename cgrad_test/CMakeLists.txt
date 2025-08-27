add_library(cgrad_test STATIC
    src/datastructures/test_list.c
    src/datastructures/test_list_callbacks.c
    src/run_tests.c
)

target_include_directories(cgrad_test PUBLIC
    ${CMAKE_SOURCE_DIR}/cgrad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cgrad_test PUBLIC
    cgrad
)
