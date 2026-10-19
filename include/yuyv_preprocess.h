#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper function to launch the CUDA kernel from C++
void launch_yuyv_preprocess(
    const void* d_src_yuyv,
    float* d_dst_rgb,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif