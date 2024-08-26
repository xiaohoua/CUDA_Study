#include <cuda.h>
#include <bits/stdc++.h>
#include "cuda_runtime.h"

// 实现fp32的fused biasadd mask scale and add的融合算子
// biasadd + mask + scale + elemwise_add四个算子的融合
// （x + bias） * mask * scale + addend;

template<typename T>
struct MaskScaleAndElemwiseAddFunctor
{
    uint8_t* mask;
    float* add_val;
    float _scale;

    MaskScaleAndElemwiseAddFunctor(uint8_t* d_mask, T* d_add_val, float scale):
        mask(d_mask), add_val(d_add_val), _scale(scale){}
    
    __device__ T operator()(T tmp, int i){
        return tmp * static_cast<float>(static_cast<bool>(mask[i]) * _scale) +add_val[i];
    }
};
// 朴素写法：和视频上的一致
template<int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAdd(FUNCTOR functor, T * dx, T * dy, T * d_bias, const int n, const int bias_size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    // 对比于59行的读向量，此处读标量，总数为元素个数n
    for (int i = gid; i < n; i += gridDim.x * blockDim.x)
    {
        // 先加上bias
        T tmp = dx[i] + d_bias[i % bias_size];
        // 再做mask+scale+elementwiseadd
        dy[i] = functor(tmp,i);
    }
}

// 使用向量化进行存取
// template<int biasSize, typename FUNCTOR, typename T>
// __global__ void FusedBaisAddVecSmem(FUNCTOR functor, T * dx, T * dy, T * d_bias, const int n, const int bias_size)
// {
//     int id =  threadIdx.x +blockIdx.x * blockDim.x;

//     for(int i=id; i<n; i+=gridDim.x*blockDim.x){
//         dy[i] = functor((dx[i] + d_bias[i%bias_size]),i);
//     }
// }
template<int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAddVecSmem(FUNCTOR functor, T * dx, T * dy, T * d_bias, const int n, const int bias_size)
{
    int id =  threadIdx.x +blockIdx.x * blockDim.x;

    for(int i=id; i<n/4; i+=gridDim.x*blockDim.x){

        float4* x = reinterpret_cast<float4*>(dx);
        float4* y = reinterpret_cast<float4*>(dy);
        y[i].x = functor((x[i].x + d_bias[(i*4)%bias_size]),i*4);
        y[i].y = functor((x[i].y + d_bias[(i*4+1)%bias_size]),i*4+1);
        y[i].z = functor((x[i].z + d_bias[(i*4+2)%bias_size]),i*4+2);
        y[i].w = functor((x[i].w + d_bias[(i*4+3)%bias_size]),i*4+3);

        // float4 x = reinterpret_cast<float4*>(dx)[i];
        // float4 y;
        // y.x = functor((x.x + d_bias[(i*4)%bias_size]),i*4);
        // y.y = functor((x.y + d_bias[(i*4+1)%bias_size]),(i*4+1));
        // y.z = functor((x.z + d_bias[(i*4+2)%bias_size]),i*4+2);
        // y.w = functor((x.w + d_bias[(i*4+3)%bias_size]),i*4+3);

        // reinterpret_cast<float4*>(dy)[i] = y;
    }
}

bool CheckRight(float * y, float * groudTruth, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (y[i] != groudTruth[i])
        {
            printf("y[%d] : %f \n", i, y[i]);
            printf("groundTruth[%d] : %f\n", i, groudTruth[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int n = 10000000;
    constexpr int bias_size = 10;
    
    float scale = 0.5;
    uint8_t * mask_tensor = new uint8_t[n];
    float * add_val = new float[n];
    // 初始化
    for (int i = 0; i < n; ++i)
    {
        mask_tensor[i] = (uint8_t)(i);
        add_val[i] = (float)(i);
    }

    float * x = (float *)malloc(sizeof(float) * n);
    float * y = (float *)malloc(sizeof(float) * n);
    float * bias = (float *)malloc(sizeof(float) * bias_size);
    for (int i = 0; i < n; ++i)
    {
        x[i] = (float)(i);
        y[i] = 0.0f;
    }
    for (int i = 0; i < bias_size; ++i)
        bias[i] = i;

    float * groudTruth = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i)
    {
        groudTruth[i] = (x[i] + bias[i % bias_size]) * static_cast<float>(static_cast<bool>(mask_tensor[i]) * scale) + add_val[i];
    }

    float * dx, * dy, * d_bias;
    cudaMalloc((void **)&dx, sizeof(float) * n);
    cudaMalloc((void **)&dy, sizeof(float) * n);
    cudaMalloc((void **)&d_bias, sizeof(float) * bias_size);
    cudaMemcpy(dx, x, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * bias_size, cudaMemcpyHostToDevice);
    uint8_t * d_mask_tensor;
    float * d_add_val;
    cudaMalloc((void **)&d_mask_tensor, sizeof(uint8_t) * n);
    cudaMalloc((void **)&d_add_val, sizeof(float) * n);
    cudaMemcpy(d_mask_tensor, mask_tensor, sizeof(uint8_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_add_val, add_val, sizeof(float) * n, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = 512;
    int gridSize = std::min((n + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);

    MaskScaleAndElemwiseAddFunctor<float> functor(d_mask_tensor, d_add_val, scale);

    dim3 Block(blockSize);
    dim3 Grid(gridSize);

    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < 1000; ++i)
        // FusedBaisAdd<bias_size><<<Grid, Block>>>(functor, dx, dy, d_bias, n, bias_size);
        FusedBaisAddVecSmem<bias_size><<<Grid, Block>>>(functor, dx, dy, d_bias, n, bias_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(y, dy, sizeof(float) * n, cudaMemcpyDeviceToHost);

    bool isRight = CheckRight(y, groudTruth, n);
    if (isRight)
        printf("结果正确\n");
    else
        printf("结果错误\n");    

    printf("it costs %f s \n", milliseconds/1000);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(d_bias);
    cudaFree(d_add_val);
    cudaFree(d_mask_tensor);
    free(x);
    free(y);
    free(bias);
    free(groudTruth);
    delete mask_tensor;
    mask_tensor = nullptr;
    delete add_val;
    add_val = nullptr;
}