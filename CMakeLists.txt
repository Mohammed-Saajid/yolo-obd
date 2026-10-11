cmake_minimum_required(VERSION 3.20)
project(InferenceBackend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV

find_package(OpenCV 5.0 REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")

# Explicitly Define Triton Paths based on your download
set(TRITON_CLIENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton/include)
set(TRITON_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton/lib)

add_library(stage_stats STATIC
    src/stage_stats.cpp
)

target_include_directories(stage_stats PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the executable and all source files
add_executable(InferenceBackend
    src/main.cpp
    src/inference_client.cpp
    src/scoped_timer.cpp
)

# Tell the compiler exactly where to find EVERY header file
target_include_directories(InferenceBackend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(InferenceBackend PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(InferenceBackend PRIVATE ${TRITON_CLIENT_DIR})

# Professional Compiler flags
target_compile_options(InferenceBackend PRIVATE -Wall -Wextra -Wpedantic)

# Tell the linker exactly where the .so files live
target_link_directories(InferenceBackend PRIVATE ${TRITON_LIB_DIR})

# Bind the compiled libraries to your executable
target_link_libraries(InferenceBackend PRIVATE 
    ${OpenCV_LIBS}
    grpcclient
    stage_stats
)

add_executable(timing_benchmark
    benchmarks/timing_benchmark.cpp
    src/inference_client.cpp
    src/scoped_timer.cpp
    src/stage_stats.cpp
)

target_include_directories(timing_benchmark PRIVATE include)
target_include_directories(timing_benchmark PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(timing_benchmark PRIVATE ${TRITON_CLIENT_DIR})
target_compile_features(timing_benchmark PRIVATE cxx_std_17)
target_compile_options(timing_benchmark PRIVATE -Wall -Wextra -Wpedantic)
target_link_directories(timing_benchmark PRIVATE ${TRITON_LIB_DIR})
target_link_libraries(timing_benchmark PRIVATE
    ${OpenCV_LIBS}
    grpcclient
)

enable_testing()
add_subdirectory(tests)