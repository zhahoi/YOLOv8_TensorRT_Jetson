#include "postprocess.h"

// ===== decode kernel for segmentation: 输出直接为原图坐标 + mask特征 =====
__global__ void decode_yolov8_seg_kernel(
    int num_class, int num_anchors, int seg_channels, float conf_thresh,
    const float* src, float* out_boxes, float* out_mask_feats, int* out_count, 
    const PreParam pparam)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    const float* psrc = src + idx * (num_class + 4 + seg_channels);
    float cx = psrc[0];
    float cy = psrc[1];
    float w  = psrc[2];
    float h  = psrc[3];

    const float* score_ptr = psrc + 4;
    float score = score_ptr[0];
    int label = 0;
    for (int i = 1; i < num_class; ++i) {
        if (score_ptr[i] > score) {
            score = score_ptr[i];
            label = i;
        }
    }
    if (score < conf_thresh) return;

    // --- 恢复到原图坐标 ---
    float x0 = (cx - 0.5f*w - pparam.dw) * pparam.ratio;
    float y0 = (cy - 0.5f*h - pparam.dh) * pparam.ratio;
    float x1 = (cx + 0.5f*w - pparam.dw) * pparam.ratio;
    float y1 = (cy + 0.5f*h - pparam.dh) * pparam.ratio;

    // 裁剪到图像边界
    x0 = fminf(fmaxf(x0, 0.f), pparam.width);
    y0 = fminf(fmaxf(y0, 0.f), pparam.height);
    x1 = fminf(fmaxf(x1, 0.f), pparam.width);
    y1 = fminf(fmaxf(y1, 0.f), pparam.height);

    if (x0 >= x1 || y0 >= y1) return;

    int index = atomicAdd(out_count, 1);
    if (index >= num_anchors) return;

    // 保存检测框信息
    int base = index * 6;
    out_boxes[base + 0] = x0;
    out_boxes[base + 1] = y0;
    out_boxes[base + 2] = x1;
    out_boxes[base + 3] = y1;
    out_boxes[base + 4] = score;
    out_boxes[base + 5] = float(label);

    // 保存mask特征
    const float* mask_feat_ptr = psrc + 4 + num_class;
    for (int i = 0; i < seg_channels; ++i) {
        out_mask_feats[index * seg_channels + i] = mask_feat_ptr[i];
    }
}

// ===== NMS kernel for segmentation =====
__global__ void nms_seg_kernel(
    const float* bboxes, const float* mask_feats, 
    int* keep_flags, int num_boxes, float iou_thresh, int seg_channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;
    keep_flags[i] = 1;

    const float* box_i = bboxes + i*6;
    for (int j = 0; j < i; ++j) {
        if (keep_flags[j] == 0) continue;
        const float* box_j = bboxes + j*6;

        if (fabs(box_i[5]-box_j[5]) > 1e-3f) continue;

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
        if (iou > iou_thresh) { keep_flags[i]=0; return; }
    }
}

// ===== Transpose kernel for segmentation =====
__global__ void transpose_yolov8_seg_kernel(
    const float* input, float* output,
    int channels, int anchors)
{
    int anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor_idx >= anchors) return;
    for (int c = 0; c < channels; ++c)
        output[anchor_idx * channels + c] = input[c * anchors + anchor_idx];
}

// ===== Matrix multiplication kernel: mask_feats * protos =====
__global__ void matmul_kernel(
    const float* mask_feats, const float* protos, float* output,
    int num_objs, int seg_channels, int seg_hw)
{
    int obj_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (obj_idx >= num_objs || pixel_idx >= seg_hw) return;
    
    float sum = 0.0f;
    for (int c = 0; c < seg_channels; ++c) {
        sum += mask_feats[obj_idx * seg_channels + c] * protos[c * seg_hw + pixel_idx];
    }
    output[obj_idx * seg_hw + pixel_idx] = sum;
}

// ===== Sigmoid kernel =====
__global__ void sigmoid_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

// ===== ROI crop and resize kernel =====
__global__ void crop_resize_kernel(
    const float* src_mask, float* dst_mask,
    int src_h, int src_w, int dst_h, int dst_w,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int obj_idx, int num_objs)
{
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dst_x >= dst_w || dst_y >= dst_h) return;
    
    // 从ROI区域映射到原mask坐标
    float src_x = roi_x + (dst_x * roi_w) / (float)dst_w;
    float src_y = roi_y + (dst_y * roi_h) / (float)dst_h;
    
    // 双线性插值
    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    x0 = max(0, min(x0, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    x1 = max(0, min(x1, src_w - 1));
    y1 = max(0, min(y1, src_h - 1));
    
    float wx = src_x - x0;
    float wy = src_y - y0;
    
    int src_base = obj_idx * src_h * src_w;
    float v00 = src_mask[src_base + y0 * src_w + x0];
    float v01 = src_mask[src_base + y0 * src_w + x1];
    float v10 = src_mask[src_base + y1 * src_w + x0];
    float v11 = src_mask[src_base + y1 * src_w + x1];
    
    float val = v00 * (1-wx) * (1-wy) + v01 * wx * (1-wy) + 
                v10 * (1-wx) * wy + v11 * wx * wy;
    
    int dst_base = obj_idx * dst_h * dst_w;
    dst_mask[dst_base + dst_y * dst_w + dst_x] = val;
}

// ===== Box mask extraction kernel =====
__global__ void extract_box_mask_kernel(
    const float* full_mask, unsigned char* box_mask,
    const float* boxes, int num_objs, int img_h, int img_w, float mask_thresh)
{
    int obj_idx = blockIdx.x;
    int pixel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (obj_idx >= num_objs) return;
    
    const float* box = boxes + obj_idx * 6;
    int box_x = max(0, min((int)box[0], img_w - 1));
    int box_y = max(0, min((int)box[1], img_h - 1));
    int box_x2 = max(0, min((int)box[2], img_w));
    int box_y2 = max(0, min((int)box[3], img_h));
    int box_w = box_x2 - box_x;
    int box_h = box_y2 - box_y;
    
    if (box_w <= 0 || box_h <= 0 || pixel_idx >= box_w * box_h) return;
    
    int local_y = pixel_idx / box_w;
    int local_x = pixel_idx % box_w;
    int global_x = box_x + local_x;
    int global_y = box_y + local_y;
    
    if (global_x >= img_w || global_y >= img_h) return;
    
    int full_mask_idx = obj_idx * img_h * img_w + global_y * img_w + global_x;
    int box_mask_idx = obj_idx * box_h * box_w + local_y * box_w + local_x;
    
    box_mask[box_mask_idx] = (full_mask[full_mask_idx] > mask_thresh) ? 255 : 0;
}

// ===== CUDA 分割后处理主函数 =====
void cuda_seg_postprocess(
    std::vector<Object>& objs,
    const float* d_det_output,    
    const float* d_proto_output,  
    int num_classes, int num_anchors, int seg_channels, int seg_h, int seg_w,
    const PreParam& pparam, float score_thresh, float iou_thresh, int topk
)
{
    objs.clear();

    int BLOCK = 256;
    int grid = (num_anchors + BLOCK - 1) / BLOCK;
    
    // 1. 转置检测输出
    float* d_trans = nullptr;
    cudaMalloc(&d_trans, num_anchors * (num_classes + 4 + seg_channels) * sizeof(float));
    transpose_yolov8_seg_kernel<<<grid, BLOCK>>>(
        d_det_output, d_trans, num_classes + 4 + seg_channels, num_anchors);
    cudaDeviceSynchronize();

    // 2. decode: 输出检测框和mask特征
    float* d_boxes = nullptr;
    float* d_mask_feats = nullptr;
    int* d_count = nullptr;
    cudaMalloc(&d_boxes, num_anchors * 6 * sizeof(float));
    cudaMalloc(&d_mask_feats, num_anchors * seg_channels * sizeof(float));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    decode_yolov8_seg_kernel<<<grid, BLOCK>>>(
        num_classes, num_anchors, seg_channels, score_thresh,
        d_trans, d_boxes, d_mask_feats, d_count, pparam);
    cudaDeviceSynchronize();

    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_count <= 0) {
        cudaFree(d_trans); cudaFree(d_boxes); cudaFree(d_mask_feats); cudaFree(d_count);
        return;
    }
    h_count = std::min(h_count, num_anchors);

    // 3. NMS
    int* d_keep = nullptr;
    cudaMalloc(&d_keep, h_count * sizeof(int));
    grid = (h_count + BLOCK - 1) / BLOCK;
    nms_seg_kernel<<<grid, BLOCK>>>(
        d_boxes, d_mask_feats, d_keep, h_count, iou_thresh, seg_channels);
    cudaDeviceSynchronize();

    // 4. 获取保留的对象
    std::vector<int> h_keep(h_count);
    std::vector<float> h_boxes(h_count * 6);
    std::vector<float> h_mask_feats(h_count * seg_channels);
    cudaMemcpy(h_keep.data(), d_keep, h_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boxes.data(), d_boxes, h_count * 6 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask_feats.data(), d_mask_feats, h_count * seg_channels * sizeof(float), cudaMemcpyDeviceToHost);

    // 统计保留的对象数量
    int kept_count = 0;
    std::vector<int> kept_indices;
    for (int i = 0; i < h_count; ++i) {
        if (h_keep[i] && kept_count < topk) {
            kept_indices.push_back(i);
            kept_count++;
        }
    }

    if (kept_count == 0) {
        cudaFree(d_trans); cudaFree(d_boxes); cudaFree(d_mask_feats); 
        cudaFree(d_count); cudaFree(d_keep);
        return;
    }

    // 5. 准备mask计算数据
    float* d_kept_mask_feats = nullptr;
    cudaMalloc(&d_kept_mask_feats, kept_count * seg_channels * sizeof(float));
    
    // 复制保留对象的mask特征
    for (int i = 0; i < kept_count; ++i) {
        int orig_idx = kept_indices[i];
        cudaMemcpy(d_kept_mask_feats + i * seg_channels,
                   d_mask_feats + orig_idx * seg_channels,
                   seg_channels * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // 6. 矩阵乘法: mask_feats * protos
    int seg_hw = seg_h * seg_w;
    float* d_raw_masks = nullptr;
    cudaMalloc(&d_raw_masks, kept_count * seg_hw * sizeof(float));

    dim3 matmul_block(16, 16);
    dim3 matmul_grid((kept_count + matmul_block.x - 1) / matmul_block.x,
                     (seg_hw + matmul_block.y - 1) / matmul_block.y);
    
    matmul_kernel<<<matmul_grid, matmul_block>>>(
        d_kept_mask_feats, d_proto_output, d_raw_masks,
        kept_count, seg_channels, seg_hw);
    cudaDeviceSynchronize();

    // 7. Sigmoid激活
    int total_mask_pixels = kept_count * seg_hw;
    grid = (total_mask_pixels + BLOCK - 1) / BLOCK;
    sigmoid_kernel<<<grid, BLOCK>>>(d_raw_masks, total_mask_pixels);
    cudaDeviceSynchronize();

    // 8. 裁剪ROI并resize到原图尺寸
    int scale_dw = (int)(pparam.dw / pparam.width * seg_w);
    int scale_dh = (int)(pparam.dh / pparam.height * seg_h);
    int roi_w = seg_w - 2 * scale_dw;
    int roi_h = seg_h - 2 * scale_dh;

    float* d_resized_masks = nullptr;
    cudaMalloc(&d_resized_masks, kept_count * (int)pparam.height * (int)pparam.width * sizeof(float));

    dim3 resize_block(16, 16);
    for (int i = 0; i < kept_count; ++i) {
        dim3 resize_grid(((int)pparam.width + resize_block.x - 1) / resize_block.x,
                        ((int)pparam.height + resize_block.y - 1) / resize_block.y);
        crop_resize_kernel<<<resize_grid, resize_block>>>(
            d_raw_masks, d_resized_masks,
            seg_h, seg_w, (int)pparam.height, (int)pparam.width,
            scale_dw, scale_dh, roi_w, roi_h, i, kept_count);
    }
    cudaDeviceSynchronize();

    // ===== 修复：计算每个对象的实际box尺寸 =====
    std::vector<cv::Rect> actual_boxes(kept_count);
    std::vector<int> box_sizes(kept_count);
    
    for (int i = 0; i < kept_count; ++i) {
        int orig_idx = kept_indices[i];
        
        // 获取实际的边界框坐标（已经限制在图像范围内）
        int x1 = std::max(0, std::min((int)h_boxes[orig_idx * 6 + 0], (int)pparam.width - 1));
        int y1 = std::max(0, std::min((int)h_boxes[orig_idx * 6 + 1], (int)pparam.height - 1));
        int x2 = std::max(0, std::min((int)h_boxes[orig_idx * 6 + 2], (int)pparam.width));
        int y2 = std::max(0, std::min((int)h_boxes[orig_idx * 6 + 3], (int)pparam.height));
        
        int box_w = x2 - x1;
        int box_h = y2 - y1;
        
        actual_boxes[i] = cv::Rect(x1, y1, box_w, box_h);
        box_sizes[i] = box_w * box_h;
    }

    // ===== 修复：为每个对象单独分配box mask内存 =====
    std::vector<unsigned char*> d_box_masks(kept_count);
    for (int i = 0; i < kept_count; ++i) {
        if (box_sizes[i] > 0) {
            cudaMalloc(&d_box_masks[i], box_sizes[i] * sizeof(unsigned char));
        } else {
            d_box_masks[i] = nullptr;
        }
    }

    // 9. 提取每个对象的box区域mask
    for (int i = 0; i < kept_count; ++i) {
        if (d_box_masks[i] == nullptr || box_sizes[i] <= 0) continue;
        
        int orig_idx = kept_indices[i];
        
        // 更新boxes数组为实际坐标
        float temp_box[6] = {
            (float)actual_boxes[i].x,
            (float)actual_boxes[i].y, 
            (float)(actual_boxes[i].x + actual_boxes[i].width),
            (float)(actual_boxes[i].y + actual_boxes[i].height),
            h_boxes[orig_idx * 6 + 4], // confidence
            h_boxes[orig_idx * 6 + 5]  // class
        };
        
        float* d_temp_box;
        cudaMalloc(&d_temp_box, 6 * sizeof(float));
        cudaMemcpy(d_temp_box, temp_box, 6 * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 extract_block(1, std::min(256, box_sizes[i]));
        dim3 extract_grid(1, (box_sizes[i] + extract_block.y - 1) / extract_block.y);
        
        // 单独处理每个对象
        extract_box_mask_kernel<<<extract_grid, extract_block>>>(
            d_resized_masks + i * (int)pparam.height * (int)pparam.width,
            d_box_masks[i],
            d_temp_box,
            1, // 只处理一个对象
            (int)pparam.height,
            (int)pparam.width,
            0.5f);
            
        cudaFree(d_temp_box);
    }
    cudaDeviceSynchronize();

    // 10. 构造最终结果
    for (int i = 0; i < kept_count; ++i) {
        int orig_idx = kept_indices[i];
        Object obj;
        
        // 使用实际的边界框
        obj.rect = cv::Rect_<float>(
            actual_boxes[i].x, actual_boxes[i].y,
            actual_boxes[i].width, actual_boxes[i].height
        );
        obj.prob = h_boxes[orig_idx * 6 + 4];
        obj.label = int(h_boxes[orig_idx * 6 + 5]);

        // 创建box mask
        if (d_box_masks[i] != nullptr && actual_boxes[i].width > 0 && actual_boxes[i].height > 0) {
            obj.boxMask = cv::Mat(actual_boxes[i].height, actual_boxes[i].width, CV_8UC1);
            
            // 从GPU复制mask数据
            cudaMemcpy(obj.boxMask.data, d_box_masks[i],
                      actual_boxes[i].width * actual_boxes[i].height * sizeof(unsigned char), 
                      cudaMemcpyDeviceToHost);
                      
            // 检查mask是否有效（调试用）
            cv::Scalar sum_val = cv::sum(obj.boxMask);
            if (sum_val[0] == 0) {
                std::cout << "Warning: Empty mask for object " << i << std::endl;
            }
        } else {
            obj.boxMask = cv::Mat(); // 创建空掩码
        }

        objs.push_back(obj);
    }

    // 按概率排序
    std::sort(objs.begin(), objs.end(), 
              [](const Object& a, const Object& b) { return a.prob > b.prob; });

    // 清理GPU内存
    cudaFree(d_trans);
    cudaFree(d_boxes);
    cudaFree(d_mask_feats);
    cudaFree(d_count);
    cudaFree(d_keep);
    cudaFree(d_kept_mask_feats);
    cudaFree(d_raw_masks);
    cudaFree(d_resized_masks);
    
    // 释放每个对象的box mask内存
    for (int i = 0; i < kept_count; ++i) {
        if (d_box_masks[i] != nullptr) {
            cudaFree(d_box_masks[i]);
        }
    }
}