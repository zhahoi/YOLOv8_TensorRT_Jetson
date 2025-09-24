#include "preprocess.h"

__global__ void centercrop_resize_kernel(
    const uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_width, int dst_height, 
    uint8_t const_value_st, AffineMatrix d2s, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx = position % dst_width;
    int dy = position / dst_width;

    // 应用仿射变换：从目标坐标映射到源坐标
    float src_x = d2s.value[0] * dx + d2s.value[1] * dy + d2s.value[2] + 0.5f;
    float src_y = d2s.value[3] * dx + d2s.value[4] * dy + d2s.value[5] + 0.5f;

    float c0, c1, c2;
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        // 超出边界，使用常数值填充
        c0 = c1 = c2 = const_value_st;
    } else {
        // 双线性插值
        int x_low = floorf(src_x);
        int y_low = floorf(src_y);
        int x_high = min(x_low + 1, src_width - 1);
        int y_high = min(y_low + 1, src_height - 1);

        const uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
        const uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
        const uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
        const uint8_t* v4 = src + y_high * src_line_size + x_high * 3;

        float lx = src_x - x_low;
        float ly = src_y - y_low;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        float w1 = hx * hy;
        float w2 = lx * hy;
        float w3 = hx * ly;
        float w4 = lx * ly;

        // BGR 插值
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // BGR -> RGB 转换
    float tmp = c0; c0 = c2; c2 = tmp;

    // 归一化到 [0,1]
    c0 /= 255.0f; c1 /= 255.0f; c2 /= 255.0f;

    // 输出到 CHW 格式: [C, H, W]
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;        // R channel
    float* pdst_c1 = pdst_c0 + area;                   // G channel
    float* pdst_c2 = pdst_c1 + area;                   // B channel

    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void cuda_preprocess(
    const uint8_t* src_host, int src_width, int src_height,
    float* dst_device, int dst_width, int dst_height,
    cudaStream_t stream)
{
    if (!src_host || !dst_device) {
        std::cerr << "[ERROR] cuda_preprocess_cls: null pointer" << std::endl;
        return;
    }

    // 1. 计算center crop + resize的仿射变换矩阵
    // 分类任务通常使用center crop：保持宽高比，裁剪较长边到正方形，然后resize
    float scale = std::max(dst_width / (float)src_width, dst_height / (float)src_height);
    
    // 计算裁剪后的尺寸
    float crop_width = dst_width / scale;
    float crop_height = dst_height / scale;
    
    // 计算裁剪的起始位置（居中裁剪）
    float crop_x = (src_width - crop_width) / 2.0f;
    float crop_y = (src_height - crop_height) / 2.0f;
    
    // 构建从源图像到目标图像的仿射变换矩阵 (src -> dst)
    AffineMatrix s2d, d2s;
    s2d.value[0] = scale;     // x缩放
    s2d.value[1] = 0.0f;      // x倾斜
    s2d.value[2] = -crop_x * scale;  // x平移
    s2d.value[3] = 0.0f;      // y倾斜
    s2d.value[4] = scale;     // y缩放
    s2d.value[5] = -crop_y * scale;  // y平移
    
    // 计算逆变换矩阵 (dst -> src)，用于kernel中的坐标映射
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    // 2. 分配GPU临时缓冲区存储原始图像
    size_t src_bytes = size_t(src_width) * size_t(src_height) * 3;
    uint8_t* d_src = nullptr;

#if defined(__CUDACC__) && (CUDA_VERSION >= 11010)
    // 优先使用异步内存分配
    cudaError_t e = cudaMallocAsync((void**)&d_src, src_bytes, stream);
    if (e != cudaSuccess) {
        // 回退到同步分配
        e = cudaMalloc((void**)&d_src, src_bytes);
    }
#else
    cudaError_t e = cudaMalloc((void**)&d_src, src_bytes);
#endif

    if (e != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc failed in cuda_preprocess_cls: " 
                  << cudaGetErrorString(e) << std::endl;
        return;
    }

    // 3. 拷贝数据：主机 -> 设备
    cudaError_t cpyErr = cudaMemcpyAsync(d_src, src_host, src_bytes, 
                                         cudaMemcpyHostToDevice, stream);
    if (cpyErr != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpyAsync H2D failed in cuda_preprocess_cls: " 
                  << cudaGetErrorString(cpyErr) << std::endl;
        
        // 清理已分配的内存
#if defined(__CUDACC__) && (CUDA_VERSION >= 11010)
        cudaFreeAsync(d_src, stream);
#else
        cudaFree(d_src);
#endif
        return;
    }

    // 4. 启动CUDA kernel
    int jobs = dst_width * dst_height;
    int threads = 256;
    int blocks = (jobs + threads - 1) / threads;
    
    centercrop_resize_kernel<<<blocks, threads, 0, stream>>>(
        d_src, src_width * 3, src_width, src_height,
        dst_device, dst_width, dst_height, 128, d2s, jobs);

    // 检查kernel启动错误
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "[ERROR] Kernel launch failed in cuda_preprocess_cls: " 
                  << cudaGetErrorString(kernelErr) << std::endl;
    }

    // 5. 同步流并释放临时缓冲区
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamSynchronize failed in cuda_preprocess_cls: " 
                  << cudaGetErrorString(syncErr) << std::endl;
    }

    // 释放临时设备内存
#if defined(__CUDACC__) && (CUDA_VERSION >= 11010)
    cudaFreeAsync(d_src, stream);
#else
    cudaFree(d_src);
#endif
}