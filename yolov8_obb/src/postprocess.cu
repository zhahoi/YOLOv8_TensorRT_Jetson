#include "postprocess.h"

// ===== decode kernel for YOLOv8 OBB =====
__global__ void decode_yolov8_obb_kernel(
    int num_anchors, int num_labels, float conf_thresh,
    const float* src, float* out_boxes, int* out_count, const PreParam pparam)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    const int num_channels = 4 + num_labels + 1; // 4(bbox) + num_labels(classes) + 1(angle)
    const float* psrc = src + idx * num_channels;
    
    // 解析边界框信息
    float cx = psrc[0];
    float cy = psrc[1];
    float w  = psrc[2];
    float h  = psrc[3];
    
    // 找到最大置信度和对应类别
    const float* scores_ptr = psrc + 4;
    float max_score = scores_ptr[0];
    int max_label = 0;
    
    for (int i = 1; i < num_labels; i++) {
        if (scores_ptr[i] > max_score) {
            max_score = scores_ptr[i];
            max_label = i;
        }
    }
    
    if (max_score < conf_thresh) return;
    
    // 获取角度信息
    float angle = psrc[4 + num_labels]; // 角度在最后
    
    // 坐标变换：与CPU版本保持一致
    float x = (cx - pparam.dw) * pparam.ratio;
    float y = (cy - pparam.dh) * pparam.ratio;
    w = w * pparam.ratio;
    h = h * pparam.ratio;
    
    // 基本验证
    if (w < 1.0f || h < 1.0f) return;
    
    // 裁剪到图像边界
    x = fminf(fmaxf(x, 0.f), pparam.width);
    y = fminf(fmaxf(y, 0.f), pparam.height);
    w = fminf(fmaxf(w, 0.f), pparam.width);
    h = fminf(fmaxf(h, 0.f), pparam.height);
    
    // 角度转换：从弧度转为度
    angle = angle / 3.14159265f * 180.0f;

    int index = atomicAdd(out_count, 1);
    if (index >= num_anchors) return;

    // 存储旋转边界框信息 (7个值: cx, cy, w, h, angle, score, label)
    int base = index * 7;
    out_boxes[base + 0] = x;        // center_x
    out_boxes[base + 1] = y;        // center_y
    out_boxes[base + 2] = w;        // width
    out_boxes[base + 3] = h;        // height
    out_boxes[base + 4] = angle;    // angle in degrees
    out_boxes[base + 5] = max_score; // confidence
    out_boxes[base + 6] = float(max_label); // class label
}

// ===== 旋转矩形IOU计算 =====
__device__ float rotated_iou(
    float cx1, float cy1, float w1, float h1, float angle1,
    float cx2, float cy2, float w2, float h2, float angle2)
{
    // 简化版旋转IOU计算
    // 这里使用近似方法：如果角度差异较大，降低IOU；否则使用标准矩形IOU
    float angle_diff = fabs(angle1 - angle2);
    if (angle_diff > 180.0f) angle_diff = 360.0f - angle_diff;
    
    // 角度差异太大时，直接返回0
    if (angle_diff > 45.0f) return 0.0f;
    
    // 计算边界框（近似为轴对齐）
    float x1_min = cx1 - w1 * 0.5f;
    float y1_min = cy1 - h1 * 0.5f;
    float x1_max = cx1 + w1 * 0.5f;
    float y1_max = cy1 + h1 * 0.5f;
    
    float x2_min = cx2 - w2 * 0.5f;
    float y2_min = cy2 - h2 * 0.5f;
    float x2_max = cx2 + w2 * 0.5f;
    float y2_max = cy2 + h2 * 0.5f;
    
    // 计算交集
    float inter_x_min = fmaxf(x1_min, x2_min);
    float inter_y_min = fmaxf(y1_min, y2_min);
    float inter_x_max = fminf(x1_max, x2_max);
    float inter_y_max = fminf(y1_max, y2_max);
    
    float inter_w = fmaxf(0.0f, inter_x_max - inter_x_min);
    float inter_h = fmaxf(0.0f, inter_y_max - inter_y_min);
    float inter_area = inter_w * inter_h;
    
    // 计算并集
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - inter_area;
    
    // 根据角度差异调整IOU
    float iou = inter_area / (union_area + 1e-6f);
    float angle_factor = 1.0f - (angle_diff / 45.0f) * 0.3f; // 角度差异越大，IOU越小
    
    return iou * angle_factor;
}

// ===== NMS kernel for OBB =====
__global__ void nms_obb_kernel(
    const float* bboxes, int* keep_flags, 
    int num_boxes, float iou_thresh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;
    keep_flags[i] = 1;

    const float* box_i = bboxes + i * 7;
    for (int j = 0; j < i; ++j) {
        if (keep_flags[j] == 0) continue;
        const float* box_j = bboxes + j * 7;

        // 检查是否为同一类别
        if (fabs(box_i[6] - box_j[6]) > 1e-3f) continue;

        // 计算旋转IOU
        float iou = rotated_iou(
            box_i[0], box_i[1], box_i[2], box_i[3], box_i[4], // box i
            box_j[0], box_j[1], box_j[2], box_j[3], box_j[4]  // box j
        );
        
        if (iou > iou_thresh) { 
            keep_flags[i] = 0; 
            return; 
        }
    }
}

// ===== Transpose kernel for OBB =====
__global__ void transpose_yolov8_obb_kernel(
    const float* input, float* output,
    int channels, int anchors)
{
    int anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor_idx >= anchors) return;
    
    for (int c = 0; c < channels; ++c) {
        output[anchor_idx * channels + c] = input[c * anchors + anchor_idx];
    }
}

// ===== CUDA 后处理主函数 for OBB =====
void cuda_postprocess(
    std::vector<Object>& objs, const float* d_output, int num_channels, int num_anchors,
    const PreParam& pparam, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    
    printf("[OBB CUDA Debug] Input params: channels=%d, anchors=%d, labels=%d, score_thresh=%.3f\n", 
           num_channels, num_anchors, num_labels, score_thres);
    printf("[OBB CUDA Debug] PreParam: dw=%.2f, dh=%.2f, ratio=%.3f, size=%.0fx%.0f\n", 
           pparam.dw, pparam.dh, pparam.ratio, pparam.width, pparam.height);
    
    int BLOCK = 256;
    int grid = (num_anchors + BLOCK - 1) / BLOCK;

    // 1. 转置输出: (20, 21504) -> (21504, 20)
    float* d_trans = nullptr;
    cudaMalloc(&d_trans, num_anchors * num_channels * sizeof(float));
    transpose_yolov8_obb_kernel<<<grid, BLOCK>>>(d_output, d_trans, num_channels, num_anchors);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[OBB CUDA Error] Transpose kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_trans);
        return;
    }

    // 2. decode: 解码旋转边界框
    float* d_boxes = nullptr;
    int* d_count = nullptr;
    
    cudaMalloc(&d_boxes, num_anchors * 7 * sizeof(float)); // 7 values per rotated box
    cudaMalloc(&d_count, sizeof(int));
    
    cudaMemset(d_count, 0, sizeof(int));
    cudaMemset(d_boxes, 0, num_anchors * 7 * sizeof(float));

    decode_yolov8_obb_kernel<<<grid, BLOCK>>>(
        num_anchors, num_labels, score_thres, d_trans, d_boxes, d_count, pparam);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[OBB CUDA Error] Decode kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_trans);
        cudaFree(d_boxes);
        cudaFree(d_count);
        return;
    }

    // 3. 获取有效检测数量
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[OBB CUDA Debug] Raw detection count: %d\n", h_count);
    
    if (h_count <= 0) {
        printf("[OBB CUDA Debug] No detections found\n");
        cudaFree(d_trans);
        cudaFree(d_boxes);
        cudaFree(d_count);
        return;
    }
    
    h_count = std::min(h_count, num_anchors);

    // 4. NMS
    int* d_keep = nullptr;
    cudaMalloc(&d_keep, h_count * sizeof(int));
    cudaMemset(d_keep, 0, h_count * sizeof(int));
    
    grid = (h_count + BLOCK - 1) / BLOCK;
    nms_obb_kernel<<<grid, BLOCK>>>(d_boxes, d_keep, h_count, iou_thres);
    cudaDeviceSynchronize();

    // 5. 复制结果到主机
    std::vector<int> h_keep(h_count);
    std::vector<float> h_boxes(h_count * 7);
    
    cudaMemcpy(h_keep.data(), d_keep, h_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boxes.data(), d_boxes, h_count * 7 * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 构造Object结果
    for (int i = 0; i < h_count; ++i) {
        if (h_keep[i] == 0) continue;
        
        Object obj;
        
        // 设置旋转边界框
        cv::RotatedRect rotated_rect;
        rotated_rect.center.x = h_boxes[i * 7 + 0];
        rotated_rect.center.y = h_boxes[i * 7 + 1];
        rotated_rect.size.width = h_boxes[i * 7 + 2];
        rotated_rect.size.height = h_boxes[i * 7 + 3];
        rotated_rect.angle = h_boxes[i * 7 + 4];
        
        obj.rect = rotated_rect;
        obj.prob = h_boxes[i * 7 + 5];
        obj.label = int(h_boxes[i * 7 + 6]);
        
        objs.push_back(obj);
        if (objs.size() >= topk) break;
    }

    printf("[OBB CUDA Debug] Final objects after NMS: %d\n", (int)objs.size());

    // 7. 按概率排序
    std::sort(objs.begin(), objs.end(), 
        [](const Object& a, const Object& b) { 
            return a.prob > b.prob; 
        });

    // 8. 清理GPU内存
    cudaFree(d_trans);
    cudaFree(d_boxes);
    cudaFree(d_count);
    cudaFree(d_keep);
}