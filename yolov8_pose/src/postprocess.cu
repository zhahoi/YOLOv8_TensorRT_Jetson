#include "postprocess.h"

// ===== decode kernel for YOLOv8 Pose: 与CPU版本保持完全一致的坐标变换 =====
__global__ void decode_yolov8_pose_kernel(
    int num_anchors, float conf_thresh,
    const float* src, float* out_boxes, float* out_kps, int* out_count, const PreParam pparam)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    const int num_channels = 56; // 4(bbox) + 1(conf) + 17*3(keypoints)
    const float* psrc = src + idx * num_channels;
    
    float cx = psrc[0];
    float cy = psrc[1];
    float w  = psrc[2];
    float h  = psrc[3];
    float score = psrc[4]; // 目标置信度

    if (score < conf_thresh) return;

    // --- 修正：与CPU版本完全一致的坐标变换顺序 ---
    // 先减去padding，再计算边界，最后乘以ratio
    float x = cx - pparam.dw;
    float y = cy - pparam.dh;
    
    float x0 = (x - 0.5f * w) * pparam.ratio;
    float y0 = (y - 0.5f * h) * pparam.ratio;
    float x1 = (x + 0.5f * w) * pparam.ratio;
    float y1 = (y + 0.5f * h) * pparam.ratio;

    // 裁剪到图像边界（使用fminf/fmaxf相当于clamp）
    x0 = fminf(fmaxf(x0, 0.f), pparam.width);
    y0 = fminf(fmaxf(y0, 0.f), pparam.height);
    x1 = fminf(fmaxf(x1, 0.f), pparam.width);
    y1 = fminf(fmaxf(y1, 0.f), pparam.height);

    if (x0 >= x1 || y0 >= y1) return;

    int index = atomicAdd(out_count, 1);
    if (index >= num_anchors) return;

    // 存储边界框信息 (6个值: x0, y0, x1, y1, score, label)
    int box_base = index * 6;
    out_boxes[box_base + 0] = x0;
    out_boxes[box_base + 1] = y0;
    out_boxes[box_base + 2] = x1;
    out_boxes[box_base + 3] = y1;
    out_boxes[box_base + 4] = score;
    out_boxes[box_base + 5] = 0.0f; // pose只有一个类别

    // 处理关键点 (17个关键点，每个3个值: x, y, visibility)
    const float* kps_ptr = psrc + 5; // 跳过bbox和conf
    int kps_base = index * 51; // 17 * 3 = 51
    
    for (int k = 0; k < 17; k++) {
        // 修正：与CPU版本一致的关键点坐标变换
        float kps_x = (kps_ptr[3 * k] - pparam.dw) * pparam.ratio;
        float kps_y = (kps_ptr[3 * k + 1] - pparam.dh) * pparam.ratio;
        float kps_s = kps_ptr[3 * k + 2]; // visibility score

        // 裁剪关键点到图像边界
        kps_x = fminf(fmaxf(kps_x, 0.f), pparam.width);
        kps_y = fminf(fmaxf(kps_y, 0.f), pparam.height);

        out_kps[kps_base + 3 * k] = kps_x;
        out_kps[kps_base + 3 * k + 1] = kps_y;
        out_kps[kps_base + 3 * k + 2] = kps_s;
    }
}

// ===== NMS kernel for pose =====
__global__ void nms_pose_kernel(
    const float* bboxes, int* keep_flags, 
    int num_boxes, float iou_thresh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;
    keep_flags[i] = 1;

    const float* box_i = bboxes + i*6;
    for (int j = 0; j < i; ++j) {
        if (keep_flags[j] == 0) continue;
        const float* box_j = bboxes + j*6;

        // pose只有一个类别，不需要检查类别
        float left   = fmaxf(box_i[0], box_j[0]);
        float top    = fmaxf(box_i[1], box_j[1]);
        float right  = fminf(box_i[2], box_j[2]);
        float bottom = fminf(box_i[3], box_j[3]);

        float w = fmaxf(0.f, right-left);
        float h = fmaxf(0.f, bottom-top);
        float inter = w*h;

        float area_i = (box_i[2]-box_i[0])*(box_i[3]-box_i[1]);
        float area_j = (box_j[2]-box_j[0])*(box_j[3]-box_j[1]);

        float iou = inter / (area_i + area_j - inter + 1e-5f);
        if (iou > iou_thresh) { 
            keep_flags[i] = 0; 
            return; 
        }
    }
}

// ===== Transpose kernel for pose =====
__global__ void transpose_yolov8_pose_kernel(
    const float* input, float* output,
    int channels, int anchors)
{
    int anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor_idx >= anchors) return;
    
    for (int c = 0; c < channels; ++c) {
        output[anchor_idx * channels + c] = input[c * anchors + anchor_idx];
    }
}

// ===== CUDA 后处理主函数 (修复版本，增加调试输出) =====
void cuda_postprocess(
    std::vector<Object>& objs, const float* d_output, int num_channels, int num_anchors,
    const PreParam& pparam, float score_thres, float iou_thres, int topk)
{
    objs.clear();
    
    int BLOCK = 256;
    int grid = (num_anchors + BLOCK - 1) / BLOCK;

    // 1. 转置输出: (56, 8400) -> (8400, 56)
    float* d_trans = nullptr;
    cudaMalloc(&d_trans, num_anchors * num_channels * sizeof(float));
    transpose_yolov8_pose_kernel<<<grid, BLOCK>>>(d_output, d_trans, num_channels, num_anchors);
    cudaDeviceSynchronize();
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_trans);
        return;
    }

    // 2. decode: 解码边界框和关键点
    float* d_boxes = nullptr;
    float* d_keypoints = nullptr;
    int* d_count = nullptr;
    
    cudaMalloc(&d_boxes, num_anchors * 6 * sizeof(float));
    cudaMalloc(&d_keypoints, num_anchors * 51 * sizeof(float));
    cudaMalloc(&d_count, sizeof(int));
    
    cudaMemset(d_count, 0, sizeof(int));
    cudaMemset(d_boxes, 0, num_anchors * 6 * sizeof(float));
    cudaMemset(d_keypoints, 0, num_anchors * 51 * sizeof(float));

    decode_yolov8_pose_kernel<<<grid, BLOCK>>>(
        num_anchors, score_thres, d_trans, d_boxes, d_keypoints, d_count, pparam);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_trans);
        cudaFree(d_boxes);
        cudaFree(d_keypoints);
        cudaFree(d_count);
        return;
    }

    // 3. 获取有效检测数量
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_count <= 0) {
        cudaFree(d_trans);
        cudaFree(d_boxes);
        cudaFree(d_keypoints);
        cudaFree(d_count);
        return;
    }
    
    h_count = std::min(h_count, num_anchors);

    // 4. NMS
    int* d_keep = nullptr;
    cudaMalloc(&d_keep, h_count * sizeof(int));
    cudaMemset(d_keep, 0, h_count * sizeof(int));
    
    grid = (h_count + BLOCK - 1) / BLOCK;
    nms_pose_kernel<<<grid, BLOCK>>>(d_boxes, d_keep, h_count, iou_thres);
    cudaDeviceSynchronize();

    // 5. 复制结果到主机
    std::vector<int> h_keep(h_count);
    std::vector<float> h_boxes(h_count * 6);
    std::vector<float> h_keypoints(h_count * 51);
    
    cudaMemcpy(h_keep.data(), d_keep, h_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boxes.data(), d_boxes, h_count * 6 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keypoints.data(), d_keypoints, h_count * 51 * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 构造Object结果
    int kept_count = 0;
    for (int i = 0; i < h_count; ++i) {
        if (h_keep[i] == 0) continue;
        
        kept_count++;
        Object obj;
        // 设置边界框
        obj.rect = cv::Rect_<float>(
            h_boxes[i * 6 + 0],                    // x
            h_boxes[i * 6 + 1],                    // y
            h_boxes[i * 6 + 2] - h_boxes[i * 6 + 0], // width
            h_boxes[i * 6 + 3] - h_boxes[i * 6 + 1]  // height
        );
        obj.prob = h_boxes[i * 6 + 4];
        obj.label = int(h_boxes[i * 6 + 5]);
        
        // 设置关键点
        obj.kps.clear();
        obj.kps.reserve(51);
        for (int k = 0; k < 51; ++k) {
            obj.kps.push_back(h_keypoints[i * 51 + k]);
        }
        
        objs.push_back(obj);
        if (objs.size() >= topk) break;
    }
    
    // 7. 按概率排序
    std::sort(objs.begin(), objs.end(), 
        [](const Object& a, const Object& b) { 
            return a.prob > b.prob; 
        });

    // 8. 清理GPU内存
    cudaFree(d_trans);
    cudaFree(d_boxes);
    cudaFree(d_keypoints);
    cudaFree(d_count);
    cudaFree(d_keep);
}