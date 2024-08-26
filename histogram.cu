#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
//2.389ms
//tips: L68和L75为遇见的两个bug，说明了在遍历的时候需要精确传入数据量，多了或少了都可能会出现垃圾值或者cuda干脆对这种情况不处理
template <int blockSize>
__global__ void histgram(int *hist_data, int *bin_data, int N)
{
    __shared__ int cache[256];
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    int tid = threadIdx.x; 
    cache[tid] = 0; 
    __syncthreads();

    for(int i = gtid; i<N; i+=gridDim.x*blockSize){
        int val = hist_data[i];
        atomicAdd(&cache[hist_data[i]], 1);
    }
    __syncthreads();
    atomicAdd(&bin_data[tid], cache[tid]);
}

bool CheckResult(int *out, int* groudtruth, int N){
    for (int i = 0; i < N; i++){
        if (out[i] != groudtruth[i]) {
            printf("in checkres, out[i]=%d, gt[i]=%d\n", out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}

int main(){
    float milliseconds = 0;
    const int N = 25600000;
    int *hist = (int *)malloc(N * sizeof(int));
    int *bin = (int *)malloc(256 * sizeof(int));
    int *bin_data;
    int *hist_data;
    cudaMalloc((void **)&bin_data, 256 * sizeof(int));
    cudaMalloc((void **)&hist_data, N * sizeof(int));

    for(int i = 0; i < N; i++){
        hist[i] = i % 256;
    }

    int *groudtruth = (int *)malloc(256 * sizeof(int));;
    for(int j = 0; j < 256; j++){
        groudtruth[j] = 100000;
    }

    cudaMemcpy(hist_data, hist, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // bug1: L68的N不能传错，之前传的256，导致L19的cache[1]打印出来为0
    histgram<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    // bug2: 同bug1，L67传进去的256表示两个buffer的数据量，这个必须得精确，之前传的N，尽管只打印第1个值，但依然导致L27打印出来的值为垃圾值
    bool is_right = CheckResult(bin, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < 256; i++){
            printf("%d ", bin[i]);
        }
        printf("\n");
    }
    printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);    

    cudaFree(bin_data);
    cudaFree(hist_data);
    free(bin);
    free(hist);
}