#ifndef PREPROCESS_CLS_H
#define PREPROCESS_CLS_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

struct AffineMatrix {
    float value[6];
};

// CUDA kernel for center crop + resize preprocessing
__global__ void centercrop_resize_kernel(
    const uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_width, int dst_height, 
    uint8_t const_value_st, AffineMatrix d2s, int edge);

// Host function for CUDA preprocessing
void cuda_preprocess(
    const uint8_t* src_host, int src_width, int src_height,
    float* dst_device, int dst_width, int dst_height,
    cudaStream_t stream);

#endif