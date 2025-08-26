add_executable(tensor_operations tensor_operations.c)

target_link_libraries(tensor_operations PRIVATE
    cgrad
    cgrad_test
)

target_include_directories(tensor_operations PRIVATE ${CMAKE_SOURCE_DIR}/cgrad_test/include)