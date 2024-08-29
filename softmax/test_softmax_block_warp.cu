#include <cmath>
#include <iostream>
#include<vector>
#include<limits>

#include<cuda_runtime.h>
#include <cuda.h>
// softmax = exp(xi) / sigma(exp(xi))

#define warpSize 32


template<typename T>
void softmax_cpu(void* src, void* dst, int row, int col){

    for(int i=0; i<row; i++){
        T sum=0;

        T max = std::numeric_limits<T>::lowest();
        T* ptr_src = static_cast<T*>(src);
        T* ptr_dst = static_cast<T*>(dst);
        for(int j=0; j<col; j++){
            if(ptr_src[i * col + j] > max) max = ptr_src[i * col + j];
        }
        for(int j=0; j<col; j++){
            sum += exp(ptr_src[i * col + j] - max);
        }
        for(int j=0; j<col; j++){
            ptr_dst[i * col + j] = exp(ptr_src[i * col + j] - max)/sum;
        }
    }
}

template<typename T>
struct ReduceMaxFunctor{
    __device__ T operator()(int offset, T val){
        return max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
};

template<typename T>
struct ReduceSumFunctor{
    __device__ T operator()(int offset, T val){
        return val += __shfl_xor_sync(0xffffffff, val, offset);
    }
};

template<typename Functor, typename T>
__device__  T ReduceOp(int tid, T val){

    for(int i=warpSize/2; i>0; i>>=1){
        val = Functor()(i, val);
    }
    return val;
}
template<typename T>
__inline__ __device__ T Inf();

template<>
__inline__ __device__ float Inf<float>() {
  return 10000000000;
}

template<typename T>
__global__ void softmax_block_level_with_shfl(void* src, void* dst, int row, int col){
    //让一个block处理一行
    int tid =  threadIdx.y * blockDim.x + threadIdx.x; 
    int blockid = blockIdx.y * gridDim.x + blockIdx.x;
    int gtid =  blockid * (blockDim.x * blockDim.y) + tid; 

    int total_step = gridDim.x * blockDim.x;
    int block_step = blockDim.x * blockDim.y;

    int row_id = blockid;

    T* ptr_src = static_cast<T*>(src);
    T* ptr_dst = static_cast<T*>(dst);
    T max_val = ptr_src[0];
    T sum_val = 0;
    __shared__ T shared_max[32];
    __shared__ T shared_sum[32];

    for(int i=row_id; i<row; i+=row){
        
        if(tid>=col) return;

        
        //先把一整行的最大值规约到一个blockDim.x里  

        for(int j = tid; j<col; j+=block_step){
            max_val = max(max_val, ptr_src[i * col + j]);
        }

        max_val = ReduceOp<ReduceMaxFunctor<T>, T>(tid, max_val);

        if (tid % 32 == 0) {
            shared_max[tid / 32] = max_val;
        }
        __syncthreads();

        if (tid < 32) {
            max_val = shared_max[tid];
        }

        __syncthreads();
        max_val = ReduceOp<ReduceMaxFunctor<T>, T>(tid, max_val);
        if(tid==0){
            shared_max[0] = max_val;
        }
        __syncthreads();
        max_val = shared_max[0];

        for(int j = tid; j<col; j+=block_step){
            sum_val += exp(ptr_src[i * col + j] - max_val);
        }
        sum_val = ReduceOp<ReduceSumFunctor<T>, T>(tid, sum_val);
        
        if (tid % 32 == 0) {
            shared_sum[tid / 32] = sum_val;
        }
        __syncthreads();

        if (tid < 32) {
            sum_val = shared_sum[tid];
        }
        __syncthreads();

        
        sum_val = ReduceOp<ReduceSumFunctor<T>, T>(tid, sum_val);
        if(tid==0){
            shared_sum[0] = sum_val;
        }
        __syncthreads();
        sum_val = shared_sum[0];

        for(int j = tid; j<col; j+=block_step){
            ptr_dst[i * col + j] = exp(ptr_src[i * col + j] - max_val) / sum_val;
        }

    }
}


template<typename T>
__global__ void softmax_block_level_with_shared_memory(void* src, void* dst, int row, int col){
    //让一个block处理一行
    int tid =  threadIdx.y * blockDim.x + threadIdx.x; 
    int blockid = blockIdx.y * gridDim.x + blockIdx.x;
    int gtid =  blockid * (blockDim.x * blockDim.y) + tid; 

    int total_step = gridDim.x * blockDim.x;
    int block_step = blockDim.x * blockDim.y;

    int row_id = blockid;

    T* ptr_src = static_cast<T*>(src);
    T* ptr_dst = static_cast<T*>(dst);
    T max_val = -1e37;
    T sum_val = 0;
    

    for(int i=row_id; i<row; i+=row){
        
        if(tid>=col) return;

        // 
        __shared__ T shared_max[1024];
        __shared__ T shared_sum[1024];

        shared_max[tid] = ptr_src[i * col + tid];
        __syncthreads();
        //先把一整行的最大值规约到一个blocksize里 
        for(int j = tid; j<col; j+=block_step){
            shared_max[tid] = max(shared_max[tid], ptr_src[i * col + j]);
        }
        __syncthreads();
        for(int j = block_step/2; j>0; j>>=1){
            if(tid<j){
                shared_max[tid] = max(shared_max[tid], shared_max[tid + j]);
            }
            
            __syncthreads();
        }
        shared_sum[tid] = 0;
        __syncthreads();
        // 先把一整行的sum规约到一个blocksize里 
        for(int j = tid; j<col; j+=block_step){
            shared_sum[tid] += exp(ptr_src[i * col + j] - shared_max[0]);
            
        }
        __syncthreads();
        for (int stride = block_step / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }

        for(int j = tid; j<col; j+=block_step){
            ptr_dst[i * col + j] = exp(ptr_src[i * col + j] - shared_max[0]) / shared_sum[0];
        }

    }
}

template<typename T>
__global__ void softmax_warp_level(void* src, void* dst, int row, int col) {
    // Let each block handle one row.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockid = blockIdx.y * gridDim.x + blockIdx.x;
    int gtid = blockid * (blockDim.x * blockDim.y) + tid;

    int total_step = gridDim.x * blockDim.x;
    int block_step = blockDim.x * blockDim.y;

    int row_id = blockid;

    T* ptr_src = static_cast<T*>(src);
    T* ptr_dst = static_cast<T*>(dst);
    T max_val = -1e37;
    T sum_val = 0;

    for(int i=row_id; i<row; i+=total_step){

        // First compute the maximum value in the row.
        for (int j = tid; j < col; j += block_step) {
            max_val = max(max_val, ptr_src[i * col + j]);
        }

        for (int offset = block_step / 2; offset > 0; offset >>= 1) {
            max_val = max(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        }

        // Store the maximum value only in the first thread of the block.
        max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
        // Then compute the sum of exponentiated values.
        for (int j = tid; j < col; j += block_step) {
            sum_val += exp(ptr_src[i * col + j] - max_val);
        }

        for (int offset = block_step / 2; offset > 0; offset >>= 1) {
            sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
        }

        // Store the sum value only in the first thread of the block.
        sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);
        // Finally compute and store the softmax value.
        for (int j = tid; j < col; j += block_step) {
            ptr_dst[i * col + j] = exp(ptr_src[i * col + j] - max_val) / sum_val;
        }
    }
}

void run_test_performance(int rows, int cols){
    float start_time_v1 = 0,start_time_v2 = 0,start_time_v3 = 0;
    std::vector<float> input_data(rows * cols, 1);
    std::vector<float> output_data_cpu(rows * cols, 0);
    std::vector<float> output_data_gpu(rows * cols, 0);

    softmax_cpu<float>(input_data.data(), output_data_cpu.data(), rows, cols);

    dim3 blocks(128, 1024/128); // 每个 block 有 1024 个线程，处理一行数据
    dim3 grids(256, (rows/256 + 31) /32 *32);   // 1000 行数据，每行一个 block
    void* d_input_data, *d_output_data;
    cudaMalloc((void**)&d_input_data, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_output_data, rows*cols*sizeof(float));
    cudaMemcpy(d_input_data, input_data.data(),rows*cols*sizeof(float),cudaMemcpyHostToDevice);

    // Warmup run to fill caches and avoid initial overhead
    softmax_warp_level<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    softmax_block_level_with_shfl<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    softmax_block_level_with_shared_memory<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmax_warp_level<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v1, start, stop);
    
    cudaEventRecord(start);
    softmax_block_level_with_shfl<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v2, start, stop);
    
    // 版本3
    cudaEventRecord(start);
    softmax_block_level_with_shared_memory<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v3, start, stop);

    
    cudaMemcpy(output_data_gpu.data(),d_output_data, rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_input_data);
    cudaFree(d_output_data);


    std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;
    std::cout << "Version 1 Elapsed Time: " << start_time_v1 << " ms" << std::endl;
    std::cout << "Version 2 Elapsed Time: " << start_time_v2 << " ms" << std::endl;
    std::cout << "Version 3 Elapsed Time: " << start_time_v3 << " ms" << std::endl;
}

int main() {
    // const int rows = 1001;
    // const int cols = 2048;

    std::vector<int> rows = {100, 300, 500, 1000, 2000};
    std::vector<int> cols = {1024, 2048, 3072, 4096, 40960};

    for (int r : rows) {
        for (int c : cols) {
            run_test_performance(r, c);
        }
    }
    

    
    return 0;
}