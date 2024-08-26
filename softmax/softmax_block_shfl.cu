#include <cmath>
#include <iostream>
#include<vector>
#include<limits>

#include<cuda_runtime.h>
#include<cuda_fp16.h>
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
            sum += exp(float(ptr_src[i * col + j] - max));
        }
        for(int j=0; j<col; j++){
            ptr_dst[i * col + j] = exp(float(ptr_src[i * col + j] - max))/(float)sum;
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
__global__ void softmax_gpu(void* src, void* dst, int row, int col){
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
        if(gtid == 0){
            printf("max_val = %d\n", max_val);
        } 
        max_val = ReduceOp<ReduceMaxFunctor<T>, T>(tid, max_val);

        if (tid % 32 == 0) {
            shared_max[tid / 32] = max_val;
        }
        __syncthreads();
        if(gtid == 0){
            printf("shared_max = %d\n", shared_max[0]);
        } 
        if (tid < 32) {
            max_val = shared_max[tid];
        }
        if(gtid == 0){
            printf("max_val = %d\n", max_val);
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


int main() {
    const int rows = 1001;
    const int cols = 2050;

    std::vector<float> input_data(rows * cols, 1);
    std::vector<float> output_data_cpu(rows * cols, 0);
    std::vector<float> output_data_gpu(rows * cols, 0);

    softmax_cpu<float>(input_data.data(), output_data_cpu.data(), rows, cols);

    dim3 grids(256, (rows/256 + 31) /32 *32); //(256, 32)
    // dim3 blocks(1024, 1); //正确
    // dim3 blocks(1000, 1); //正确
    // dim3 blocks(1025, 1); //错误
    dim3 blocks(128, 8); //(128, 32)，错误//128*8=1024保证结果正确，128*9就不行

    void* d_input_data, *d_output_data;
    cudaMalloc((void**)&d_input_data, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_output_data, rows*cols*sizeof(float));
    cudaMemcpy(d_input_data, input_data.data(),rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    
    softmax_gpu<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);

    cudaMemcpy(output_data_gpu.data(),d_output_data, rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_input_data);
    cudaFree(d_output_data);

    // Check if CPU and GPU results are the same
    bool success = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(output_data_cpu[i] - output_data_gpu[i]) > 1e-5) {
            success = false;
            printf(" i =%d, output_data_cpu = %f, output_data_npu = %f\n", i, output_data_cpu[i], output_data_gpu[i]);
            break;
        }
    }

    if(success) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%lf ",output_data_gpu[i]);
        }
        printf("\n");
        for(int i=0;i<10;i++){
            printf("%lf ",output_data_cpu[i]);
        }
        printf("\n");
    }
    return 0;
}
int main() {
    const int rows = 1000;
    const int cols = 1024;

    std::vector<float> input_data(rows * cols, 1);
    std::vector<float> output_data_cpu(rows * cols, 0);
    std::vector<float> output_data_gpu(rows * cols, 0);

    softmax_cpu<float>(input_data.data(), output_data_cpu.data(), rows, cols);

    // dim3 grids(256, (rows/256 + 31) /32 *32); //(256, 32)
    dim3 grids(10,100);

    dim3 blocks(128, 8); 

    void* d_input_data, *d_output_data;
    cudaMalloc((void**)&d_input_data, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_output_data, rows*cols*sizeof(float));
    cudaMemcpy(d_input_data, input_data.data(),rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    
    softmax_gpu<float><<<grids, blocks>>>(d_input_data, d_output_data, rows, cols);

    cudaMemcpy(output_data_gpu.data(),d_output_data, rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_input_data);
    cudaFree(d_output_data);

    // Check if CPU and GPU results are the same
    bool success = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(float(output_data_cpu[i] - output_data_gpu[i])) > 1e-5) {
            success = false;
            printf(" i =%d, output_data_cpu = %f, output_data_npu = %f\n", i, output_data_cpu[i], output_data_gpu[i]);
            break;
        }
    }

    if(success) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%lf ",(float)output_data_gpu[i]);
        }
        printf("\n");
        for(int i=0;i<10;i++){
            printf("%lf ",output_data_cpu[i]);
        }
        printf("\n");
    }
    return 0;
}