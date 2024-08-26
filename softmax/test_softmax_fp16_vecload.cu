#include <cmath>
#include <iostream>
#include<vector>
#include<limits>

#include<cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
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
__global__ void softmax_block_level_fp16(void* src, void* dst, int row, int col){
    //让一个block处理一行
    int tid =  threadIdx.y * blockDim.x + threadIdx.x; 
    int blockid = blockIdx.y * gridDim.x + blockIdx.x;
    int gtid =  blockid * (blockDim.x * blockDim.y) + tid; 

    int block_step = blockDim.x * blockDim.y;

    int row_id = blockid;

    T* ptr_src = reinterpret_cast<T*>(src);
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
            shared_max[tid] = (max(__half2float(shared_max[tid]), __half2float(ptr_src[i * col + j])));
        }
        __syncthreads();
        for(int j = block_step/2; j>0; j>>=1){
            if(tid<j){
                shared_max[tid] = (max(__half2float(shared_max[tid]), __half2float(shared_max[tid + j])));
            }
            
            __syncthreads();
        }
        shared_sum[tid] = 0;
        __syncthreads();
        // 先把一整行的sum规约到一个blocksize里 
        for(int j = tid; j<col; j+=block_step){
            shared_sum[tid] += (expf(__half2float(ptr_src[i * col + j]) - __half2float(shared_max[0])));
            
        }
        __syncthreads();
        for (int stride = block_step / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }

        for(int j = tid; j<col; j+=block_step){
            ptr_dst[i * col + j] = (expf(ptr_src[i * col + j] - shared_max[0]) / __half2float(shared_sum[0]));
        }

    }
}


template<typename T>
__global__ void softmax_block_level(void* src, void* dst, int row, int col){
    //让一个block处理一行
    int tid =  threadIdx.y * blockDim.x + threadIdx.x; 
    int blockid = blockIdx.y * gridDim.x + blockIdx.x;
    int gtid =  blockid * (blockDim.x * blockDim.y) + tid; 

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
        if(gtid == 0) printf("%f\n", max_val);
        // Then compute the sum of exponentiated values.
        for (int j = tid; j < col; j += block_step) {
            sum_val += exp(ptr_src[i * col + j] - max_val);
            // if(blockid == 0)printf("%d\t", j);
        }

        for (int offset = block_step / 2; offset > 0; offset >>= 1) {
            sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
        }

        // Store the sum value only in the first thread of the block.
        sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);
        if(gtid == 0) printf("\n%f\n", sum_val);
        // Finally compute and store the softmax value.
        for (int j = tid; j < col; j += block_step) {
            ptr_dst[i * col + j] = exp(ptr_src[i * col + j] - max_val) / sum_val;
        }
    }
}
template<typename T>
__global__ void softmax_warp_level_vec(void* src, void* dst, int row, int col) {
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
    //将一行的数据规约到一个block里时，一个线程需要处理多少数据才能完成，
    //当col=2048,block_step=1024(也就是线程总数是1024时)，col_pre_thread=2
    //意味着进行float4运算时，一个线程能处理两个float4，就只需要512个（tid/2）线程工作。
    //当col=8192时，col_pre_thread=1/2，一个线程能处理1/2个float4，就需要2048个（tid/1/2）线程工作。
    int col_pre_thread = (block_step*4/col) < 1 ? 1 : (block_step*4/col);
    for(int i=row_id; i<row; i+=total_step){

        // First compute the maximum value in the row.
        for (int j = tid/col_pre_thread; j*4 < col; j += block_step) {
            float4* src_float4 = reinterpret_cast<float4*>(ptr_src + i * col);
            float4 data = src_float4[j];
            max_val = max(max_val, max(max(data.x, data.y), max(data.z, data.w)));
        }

        for (int offset = block_step / 2/col_pre_thread; offset > 0; offset >>= 1) {
            max_val = max(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        }

        // Store the maximum value only in the first thread of the block.
        max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
        // Then compute the sum of exponentiated values.
        for (int j = tid/col_pre_thread; j*4 < col; j += block_step) {
            float4* src_float4 = reinterpret_cast<float4*>(ptr_src + i * col);
            float4 data = src_float4[j];
            sum_val += exp(data.x - max_val) + exp(data.y - max_val) + exp(data.z - max_val) + exp(data.w - max_val);
        }
        for (int offset = block_step / 2/col_pre_thread; offset > 0; offset >>= 1) {
            sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
        }

        // Store the sum value only in the first thread of the block.
        sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);
        // Finally compute and store the softmax value.
        for (int j = tid; j < col/4; j += block_step) {
            float4* src_float4 = reinterpret_cast<float4*>(ptr_src + i * col);
            float4* dst_float4 = reinterpret_cast<float4*>(ptr_dst + i * col);
            float4 data = src_float4[j];
            dst_float4[j] = make_float4(exp(data.x - max_val) / sum_val, exp(data.y - max_val) / sum_val, exp(data.z - max_val) / sum_val, exp(data.w - max_val) / sum_val);
        }
    }
}

void run_test_performance(int rows, int cols){
    float start_time_v1 = 0,start_time_v2 = 0,start_time_v3 = 0,start_time_v4 = 0;
    std::vector<float> input_data(rows * cols, 1);
    std::vector<half> input_data_fp16(rows * cols*2, 1);
    std::vector<float> output_data_cpu(rows * cols, 0);
    std::vector<float> output_data_gpu(rows * cols, 0);
    std::vector<half> output_data_gpu_fp16(rows * cols*2, 0);

    softmax_cpu<float>(input_data.data(), output_data_cpu.data(), rows, cols);

    dim3 blocks(128, 1024/128); // 每个 block 有 1024 个线程，处理一行数据
    dim3 grids(256, (rows/256 + 31) /32 *32);   // 1000 行数据，每行一个 block
    void* d_input_data, *d_output_data, *d_input_data_fp16, *d_output_data_fp16;
    cudaMalloc((void**)&d_input_data, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_output_data, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_input_data_fp16, rows*cols*2*sizeof(half));
    cudaMalloc((void**)&d_output_data_fp16, rows*cols*2*sizeof(half));
    cudaMemcpy(d_input_data, input_data.data(),rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_data_fp16, input_data_fp16.data(),rows*cols*2*sizeof(half),cudaMemcpyHostToDevice);

    // Warmup run to fill caches and avoid initial overhead
    softmax_block_level<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    softmax_block_level_fp16<half><<<grids, blocks>>>(d_input_data_fp16, d_output_data_fp16, rows, cols);
    softmax_warp_level<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    softmax_warp_level_vec<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmax_block_level<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v1, start, stop);
    
    cudaEventRecord(start);
    softmax_block_level_fp16<half><<<grids, blocks>>>(d_input_data_fp16, d_output_data_fp16, rows, cols*2);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v2, start, stop);
    
    // 版本3
    cudaEventRecord(start);
    softmax_warp_level<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v3, start, stop);

    cudaEventRecord(start);
    softmax_warp_level_vec<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v4, start, stop);

    
    cudaMemcpy(output_data_gpu.data(),d_output_data, rows*cols*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(output_data_gpu_fp16.data(),d_output_data_fp16, rows*cols*2*sizeof(half),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_input_data);
    cudaFree(d_output_data);
    cudaFree(d_input_data_fp16);
    cudaFree(d_output_data_fp16);


    std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;
    std::cout << "block_level Elapsed Time: " << start_time_v1 << " ms" << std::endl;
    std::cout << "block_level_fp16 Elapsed Time: " << start_time_v2 << " ms" << std::endl;
    std::cout << "warp_level Elapsed Time: " << start_time_v3 << " ms" << std::endl;
    std::cout << "warp_level_vec Elapsed Time: " << start_time_v4 << " ms" << std::endl;
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
    
    

    
    
    // Check if CPU and GPU results are the same
    // bool success = true;
    // for (int i = 0; i < rows * cols; i++) {
    //     if (fabs(output_data_cpu[i] - output_data_gpu[i]) > 1e-5) {
    //         success = false;
    //         printf(" i =%d, output_data_cpu = %f, output_data_npu = %f\n", i, output_data_cpu[i], output_data_gpu[i]);
    //         break;
    //     }
    // }

    // if(success) {
    //     printf("the ans is right\n");
    // } else {
    //     printf("the ans is wrong\n");
    //     for(int i=0;i<10;i++){
    //         printf("%lf ",output_data_gpu[i]);
    //     }
    //     printf("\n");
    //     for(int i=0;i<10;i++){
    //         printf("%lf ",output_data_cpu[i]);
    //     }
    //     printf("\n");
    // }

    
    return 0;
}