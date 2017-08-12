cmake_minimum_required(VERSION 3.8)
project(machine_learning)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(SOURCE_FILES ${SOURCE_DIR}/main.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(MAIN_FILE ${MAIN_SOURCE_FILES})
    message(STATUS "${MAIN_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -g -fPIC -Wall -Wextra")
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include)

# compile options
# target_compile_features(${PROJECT_NAME}
#     PRIVATE
#     cxx_lambdas)

## Thread library dependency
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_THREAD_LIBS_INIT})

# install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${PROJECT_NAME}/bin)

