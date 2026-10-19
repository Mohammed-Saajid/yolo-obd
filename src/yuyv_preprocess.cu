#include "yuyv_preprocess.h"
#include <cstdint>

__global__ void yuyv2rgb_planar_kernel(
    const uint8_t* __restrict__ src_yuyv,
    float* __restrict__ dst_rgb,
    int input_width,
    int input_height,
    int output_width,
    int output_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height) return;

    const int source_x = min(x * input_width / output_width, input_width - 1);
    const int source_y = min(y * input_height / output_height, input_height - 1);
    const int frame_stride = input_width * 2;
    const int pair_x = (source_x / 2) * 4;
    const uint8_t* pixel_pair = src_yuyv + source_y * frame_stride + pair_x;

    float y_val = (source_x % 2 == 0) ? static_cast<float>(pixel_pair[0]) : static_cast<float>(pixel_pair[2]);
    float u_val = static_cast<float>(pixel_pair[1]) - 128.0f;
    float v_val = static_cast<float>(pixel_pair[3]) - 128.0f;

    float r = y_val + 1.402f * v_val;
    float g = y_val - 0.344136f * u_val - 0.714136f * v_val;
    float b = y_val + 1.772f * u_val;

    r = fminf(fmaxf(r, 0.0f), 255.0f) / 255.0f;
    g = fminf(fmaxf(g, 0.0f), 255.0f) / 255.0f;
    b = fminf(fmaxf(b, 0.0f), 255.0f) / 255.0f;

    int plane_size = output_width * output_height;
    int pixel_idx = y * output_width + x;

    dst_rgb[0 * plane_size + pixel_idx] = r;
    dst_rgb[1 * plane_size + pixel_idx] = g;
    dst_rgb[2 * plane_size + pixel_idx] = b;
}

extern "C" void launch_yuyv_preprocess(
    const void* d_src_yuyv,
    float* d_dst_rgb,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    cudaStream_t stream
) {
    dim3 block(32, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y);

    yuyv2rgb_planar_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(d_src_yuyv),
        d_dst_rgb,
        input_width,
        input_height,
        output_width,
        output_height
    );
}