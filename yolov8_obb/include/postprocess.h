#ifndef OBB_NORMAL_POSTPROCESS_CUH
#define OBB_NORMAL_POSTPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "common.hpp"

using namespace obb;

void cuda_postprocess(std::vector<Object>& objs, const float* d_output, int num_channels, int num_anchors,
    const PreParam& pparam, float score_thres, float iou_thres, int topk, int num_labels);

#endif