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
__global__ void softmax_gpu(void* src, void* dst, int row, int col) {
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