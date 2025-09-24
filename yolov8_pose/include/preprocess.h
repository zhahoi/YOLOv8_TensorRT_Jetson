#ifndef POSE_NORMAL_PREPROCESS_CUH
#define POSE_NORMAL_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cstring>

#include "common.hpp"

struct AffineMatrix {
    float value[6];
};

void cuda_preprocess(const uint8_t* src, int src_width, int src_height,
    float* dst_device, int dst_width, int dst_height,
    cudaStream_t stream, pose::PreParam& pparam);

#endif