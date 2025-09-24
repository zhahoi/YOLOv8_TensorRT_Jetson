#ifndef SEGMENT_NORMAL_POSTPROCESS_CUH
#define SEGMENT_NORMAL_POSTPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "common.hpp"

using namespace seg;

void cuda_seg_postprocess(std::vector<Object>& objs,
    const float* d_det_output,    
    const float* d_proto_output,  
    int num_classes, int num_anchors, int seg_channels, int seg_h, int seg_w,
    const PreParam& pparam, float score_thresh, float iou_thresh, int topk);

#endif