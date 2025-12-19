#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 添加必要的 half 支持头文件
#include "multiscaleDeformableAttn.h" // 确保包含您的头文件

// 测试参数配置
constexpr int batch = 1;
constexpr int spatial_size = 1000;
constexpr int num_heads = 8;
constexpr int channels = 32;
constexpr int num_levels = 4;
constexpr int num_query = 3000;
constexpr int num_point = 4;
constexpr int warmup = 10;
constexpr int iterations = 100;

// 添加 half 类型的别名定义
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    typedef __half half;
#else
    struct __half;
    typedef struct __half half;
#endif

void test_performance(bool use_fp16) {
    // 分配主机内存
    int32_t* spatialShapes = new int32_t[num_levels * 2];
    int32_t* levelStartIndex = new int32_t[num_levels];
    
    // 初始化数据（简化版）
    for (int i = 0; i < num_levels; ++i) {
        spatialShapes[i*2] = 64;   // H
        spatialShapes[i*2+1] = 64; // W
        levelStartIndex[i] = i * 4096;
    }
    
    // 分配设备内存
    void *d_value, *d_samplingLoc, *d_attnWeight, *d_output;
    int32_t *d_spatialShapes, *d_levelStartIndex;
    
    size_t value_size = batch * spatial_size * num_heads * channels * (use_fp16 ? sizeof(half) : sizeof(float));
    size_t loc_size = batch * num_query * num_heads * num_levels * num_point * 2 * (use_fp16 ? sizeof(half) : sizeof(float));
    size_t weight_size = batch * num_query * num_heads * num_levels * num_point * (use_fp16 ? sizeof(half) : sizeof(float));
    size_t output_size = batch * num_query * num_heads * channels * (use_fp16 ? sizeof(half) : sizeof(float));
    
    cudaMalloc(&d_value, value_size);
    cudaMalloc(&d_spatialShapes, num_levels * 2 * sizeof(int32_t));
    cudaMalloc(&d_levelStartIndex, num_levels * sizeof(int32_t));
    cudaMalloc(&d_samplingLoc, loc_size);
    cudaMalloc(&d_attnWeight, weight_size);
    cudaMalloc(&d_output, output_size);
    
    // 初始化设备数据
    cudaMemcpy(d_spatialShapes, spatialShapes, num_levels * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_levelStartIndex, levelStartIndex, num_levels * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    // 初始化其他张量（简化）
    cudaMemset(d_value, 0, value_size);
    cudaMemset(d_samplingLoc, 0, loc_size);
    cudaMemset(d_attnWeight, 0, weight_size);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Warm-up
    for (int i = 0; i < warmup; ++i) {
        if (use_fp16) {
            ms_deform_attn_cuda_forward(stream, 
                reinterpret_cast<half*>(d_value),
                d_spatialShapes,
                d_levelStartIndex,
                reinterpret_cast<half*>(d_samplingLoc),
                reinterpret_cast<half*>(d_attnWeight),
                reinterpret_cast<half*>(d_output),
                batch, spatial_size, num_heads, channels, 
                num_levels, num_query, num_point);
        } else {
            ms_deform_attn_cuda_forward(stream, 
                reinterpret_cast<float*>(d_value),
                d_spatialShapes,
                d_levelStartIndex,
                reinterpret_cast<float*>(d_samplingLoc),
                reinterpret_cast<float*>(d_attnWeight),
                reinterpret_cast<float*>(d_output),
                batch, spatial_size, num_heads, channels, 
                num_levels, num_query, num_point);
        }
    }
    cudaStreamSynchronize(stream);
    
    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (use_fp16) {
            ms_deform_attn_cuda_forward(stream, 
                reinterpret_cast<half*>(d_value),
                d_spatialShapes,
                d_levelStartIndex,
                reinterpret_cast<half*>(d_samplingLoc),
                reinterpret_cast<half*>(d_attnWeight),
                reinterpret_cast<half*>(d_output),
                batch, spatial_size, num_heads, channels, 
                num_levels, num_query, num_point);
        } else {
            ms_deform_attn_cuda_forward(stream, 
                reinterpret_cast<float*>(d_value),
                d_spatialShapes,
                d_levelStartIndex,
                reinterpret_cast<float*>(d_samplingLoc),
                reinterpret_cast<float*>(d_attnWeight),
                reinterpret_cast<float*>(d_output),
                batch, spatial_size, num_heads, channels, 
                num_levels, num_query, num_point);
        }
    }
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    
    // 计算耗时
    float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
    float avg_ms = total_ms / iterations;
    
    std::cout << (use_fp16 ? "FP16" : "FP32") 
              << " Mode Avg Time: " << avg_ms << " ms" 
              << " | Total (" << iterations << " runs): " << total_ms << " ms" 
              << std::endl;
    
    // 清理资源
    cudaFree(d_value);
    cudaFree(d_spatialShapes);
    cudaFree(d_levelStartIndex);
    cudaFree(d_samplingLoc);
    cudaFree(d_attnWeight);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    delete[] spatialShapes;
    delete[] levelStartIndex;
}

int main() {
    std::cout << "===== FP32 Performance Test =====" << std::endl;
    test_performance(false);
    
    std::cout << "\n===== FP16 Performance Test =====" << std::endl;
    test_performance(true);
    
    return 0;
}