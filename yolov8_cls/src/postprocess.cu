#include "postprocess.h"

// ===== 简化的ArgMax kernel for classification =====
__global__ void argmax_cls_kernel(
    const float* input, int num_classes,
    float* out_score, int* out_label)
{
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // 使用共享内存进行归约
    extern __shared__ float sdata[];
    int* s_indices = (int*)&sdata[block_size];
    
    // 初始化
    float local_max = -FLT_MAX;
    int local_index = -1;
    
    // 每个线程处理多个元素
    for (int i = tid; i < num_classes; i += block_size) {
        if (input[i] > local_max) {
            local_max = input[i];
            local_index = i;
        }
    }
    
    sdata[tid] = local_max;
    s_indices[tid] = local_index;
    __syncthreads();
    
    // 归约找到全局最大值
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                s_indices[tid] = s_indices[tid + s];
            }
        }
        __syncthreads();
    }
    
    // 线程0写入结果
    if (tid == 0) {
        *out_score = sdata[0];
        *out_label = s_indices[0];
    }
}

// ===== CUDA 后处理主函数 for Classification =====
void cuda_postprocess(
    std::vector<Object>& objs, const float* d_output, int num_classes)
{
    objs.clear();
    
    // 分配设备内存 - 只需要一个结果
    float* d_score = nullptr;
    int* d_label = nullptr;
    
    cudaMalloc(&d_score, sizeof(float));
    cudaMalloc(&d_label, sizeof(int));
    
    int BLOCK = 256;
    int shared_mem_size = BLOCK * (sizeof(float) + sizeof(int));
    
    // 启动简化的argmax kernel
    argmax_cls_kernel<<<1, BLOCK, shared_mem_size>>>(
        d_output, num_classes, d_score, d_label);
    
    cudaDeviceSynchronize();
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CLS CUDA Error] Kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_score);
        cudaFree(d_label);
        return;
    }
    
    // 复制结果到主机
    float h_score;
    int h_label;
    
    cudaMemcpy(&h_score, d_score, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_label, d_label, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 构造Object结果
    Object obj;
    obj.label = h_label;
    obj.prob = h_score;
    objs.push_back(obj);
    
    // 清理GPU内存
    cudaFree(d_score);
    cudaFree(d_label);
}
