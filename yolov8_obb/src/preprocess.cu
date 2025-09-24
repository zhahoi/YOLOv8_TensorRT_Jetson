#include "preprocess.h"


__global__ void warpaffine_kernel(
    const uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx = position % dst_width;
    int dy = position / dst_width;

    float src_x = d2s.value[0] * dx + d2s.value[1] * dy + d2s.value[2] + 0.5f;
    float src_y = d2s.value[3] * dx + d2s.value[4] * dy + d2s.value[5] + 0.5f;

    float c0, c1, c2;
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        c0 = c1 = c2 = const_value_st;
    } else {
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

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // BGR -> RGB
    float tmp = c0; c0 = c2; c2 = tmp;

    // Normalize to [0,1]
    c0 /= 255.f; c1 /= 255.f; c2 /= 255.f;

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;

    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// host-facing preprocess: copies host->device temporary buffer, runs kernel, frees
void cuda_preprocess(
    const uint8_t* src_host, int src_width, int src_height,
    float* dst_device, int dst_width, int dst_height,
    cudaStream_t stream, obb::PreParam& pparam)
{
    if (!src_host || !dst_device) {
        std::cerr << "[ERROR] cuda_preprocess null pointer" << std::endl;
        return;
    }

    // 1. 计算 letterbox 参数（与 letterbox 函数一致）
    float r = std::min(dst_width / (float)src_width, dst_height / (float)src_height);
    int new_w = std::round(src_width * r);
    int new_h = std::round(src_height * r);
    
    // 计算填充值
    float dw = (dst_width - new_w) / 2.0f;
    float dh = (dst_height - new_h) / 2.0f;
    
    // 设置预处理参数（与 letterbox 函数一致）
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.ratio = 1.0f / r;  // 存储缩放比例的倒数
    pparam.width = src_width;   // 原始图像宽度
    pparam.height = src_height; // 原始图像高度

    // 2. 计算仿射变换矩阵
    AffineMatrix s2d, d2s;
    s2d.value[0] = r; 
    s2d.value[1] = 0.f; 
    s2d.value[2] = dw; // 正确的平移值
    s2d.value[3] = 0.f; 
    s2d.value[4] = r; 
    s2d.value[5] = dh; // 正确的平移值
    
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    // 3. 分配临时设备缓冲区存储原始图像字节
    size_t src_bytes = size_t(src_width) * size_t(src_height) * 3;
    uint8_t* d_src = nullptr;
#if defined(__CUDACC__) && (CUDA_VERSION >= 11010)
    // 如果可用则使用 cudaMallocAsync，否则回退到 cudaMalloc
    cudaError_t e = cudaMallocAsync((void**)&d_src, src_bytes, stream);
    if (e != cudaSuccess) {
        // 回退
        cudaMalloc((void**)&d_src, src_bytes);
    }
#else
    cudaError_t e = cudaMalloc((void**)&d_src, src_bytes);
    if (e != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc failed: " << cudaGetErrorString(e) << std::endl;
        return;
    }
#endif

    // 4. 拷贝主机 -> 设备 (原始 BGR uint8)
    cudaError_t cpyErr = cudaMemcpyAsync(d_src, src_host, src_bytes, cudaMemcpyHostToDevice, stream);
    if (cpyErr != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpyAsync H2D failed: " << cudaGetErrorString(cpyErr) << std::endl;
#if defined(__CUDACC__) && (CUDA_VERSION >= 11010)
        cudaFreeAsync(d_src, stream);
#else
        cudaFree(d_src);
#endif
        return;
    }

    // 5. 启动内核 (dst_device 是 float* 布局 CxHxW)
    int jobs = dst_width * dst_height;
    int threads = 256;
    int blocks = (jobs + threads - 1) / threads;
    warpaffine_kernel<<<blocks, threads, 0, stream>>>(
        d_src, src_width * 3, src_width, src_height,
        dst_device, dst_width, dst_height, 128, d2s, jobs);

    // 6. 同步并释放临时缓冲区
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamSynchronize failed: " << cudaGetErrorString(syncErr) << std::endl;
    }

#if defined(__CUDACC__) && (CUDA_VERSION >= 11010)
    // 如果使用了 cudaMallocAsync，优先使用 cudaFreeAsync
    cudaFreeAsync(d_src, stream);
#else
    cudaFree(d_src);
#endif
}
