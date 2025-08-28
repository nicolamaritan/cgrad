add_executable(tensor_operations tensor_operations.c)
add_executable(tensor_allocation tensor_allocation.c)

target_link_libraries(tensor_operations PRIVATE
    cgrad
    cgrad_test
)

target_include_directories(tensor_operations PRIVATE ${CMAKE_SOURCE_DIR}/cgrad_test/include)

target_link_libraries(tensor_allocation PRIVATE
    cgrad
    cgrad_test
)

target_include_directories(tensor_allocation PRIVATE ${CMAKE_SOURCE_DIR}/cgrad_test/include)