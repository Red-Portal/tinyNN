cmake_minimum_required(VERSION 3.8)
project(machine_learning)

set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

set(blaze_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/blaze)
set(tinyNN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(${SOURCE_DIR}/perceptron)
add_subdirectory(${SOURCE_DIR}/multi_layer_perceptron)
