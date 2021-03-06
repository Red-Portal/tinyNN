cmake_minimum_required(VERSION 3.8)
project(multi_layer_perceptron)

set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/main.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)

message("blaze include: ${blaze_INCLUDE_DIR}" )

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${blaze_INCLUDE_DIR}
    ${tinyNN_INCLUDE_DIR})

## Thread library dependency
# find_package(Threads REQUIRED)

# target_link_libraries(${PROJECT_NAME}
#     PRIVATE
#     ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${PROJECT_ROOT_DIR}/bin)
