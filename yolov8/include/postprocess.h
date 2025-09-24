#ifndef DETECT_NORMAL_POSTPROCESS_CUH
#define DETECT_NORMAL_POSTPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "common.hpp"

using namespace det;

std::vector<Object> cuda_postprocess(const float* d_output, int num_classes, int num_anchors, 
    const PreParam& pparam, float score_thresh, float iou_thresh, int topk);

// 添加这个声明
__global__ void transpose_yolov8_kernel(
    const float* input, float* output, int channels, int anchors);

#endif