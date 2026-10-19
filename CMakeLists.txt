cmake_minimum_required(VERSION 3.20)
project(InferenceBackend VERSION 1.0.0 LANGUAGES CXX CUDA)

# Default to Release build if none is explicitly specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Global C++17 Standard Enforcement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV 5.0 REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# Find Cuda ToolKit
find_package(CUDAToolkit REQUIRED)



# Resolve Triton Client Library & Setup Imported Target
set(TRITON_CLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton/include")
set(TRITON_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton/lib")

find_library(TRITON_GRPC_LIB
    NAMES grpcclient
    HINTS ${TRITON_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

add_library(camera_io STATIC
    src/native_v4l2_camera.cpp
    src/yuyv_preprocess.cu
)
target_include_directories(camera_io PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(
    camera_io PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
)

add_library(Triton::grpcclient UNKNOWN IMPORTED)
set_target_properties(Triton::grpcclient PROPERTIES
    IMPORTED_LOCATION "${TRITON_GRPC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${TRITON_CLIENT_DIR}"
)

# Common Interface Target for Warning and Optimization Flags across build configurations
add_library(project_warnings INTERFACE)
target_compile_options(project_warnings INTERFACE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4
        $<$<CONFIG:Debug>:/Zi /Ob0 /Od /RTC1>
        $<$<CONFIG:Release>:/O2 /Ob2 /DNDEBUG>
    >
)

# Target 1: Internal Static Library for Timing & Statistics
add_library(stage_stats STATIC
    src/stage_stats.cpp
)
target_include_directories(stage_stats PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(stage_stats PRIVATE project_warnings)

# Target 2: Core Inference Client Static Library (Eliminates duplication between main and benchmark)
add_library(inference_core STATIC
    src/inference_client.cpp
    src/scoped_timer.cpp
)
target_compile_definitions(inference_core PUBLIC TRITON_ENABLE_GPU)
target_include_directories(inference_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(inference_core PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
    Triton::grpcclient
    stage_stats
PRIVATE
    project_warnings
)

# Target 3: Main Executable
add_executable(InferenceBackend
    src/main.cpp
)
target_link_libraries(InferenceBackend PRIVATE
    inference_core
    project_warnings
    camera_io
)

# Target 4: Timing Benchmark Executable
add_executable(timing_benchmark
    benchmarks/timing_benchmark.cpp
)
target_link_libraries(timing_benchmark PRIVATE
    inference_core
    camera_io
    ${OpenCV_LIBS}
    project_warnings
)

# Testing Framework
enable_testing()
add_subdirectory(tests)