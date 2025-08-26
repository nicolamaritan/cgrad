cmake_minimum_required(VERSION 3.10)
project(CGrad VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(cgrad)
add_subdirectory(examples)
add_subdirectory(cgrad_test)