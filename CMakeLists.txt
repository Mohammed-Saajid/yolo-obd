cmake_minimum_required(VERSION 3.20)
project(InferenceBackend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV installed via apt
find_package(OpenCV REQUIRED)

# Explicitly Define Triton Paths based on your download
set(TRITON_CLIENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton/include)
set(TRITON_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton/lib)

# Define the executable and all source files
add_executable(InferenceBackend 
    src/main.cpp
    src/inference_client.cpp
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
)