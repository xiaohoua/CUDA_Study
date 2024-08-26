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
        if(gtid == 0) printf("%f\n", max_val);
        // Then compute the sum of exponentiated values.
        for (int j = tid/col_pre_thread; j*4 < col; j += block_step) {
            if(gtid == 0) printf("\nj = %d\n", j);
            float4* src_float4 = reinterpret_cast<float4*>(ptr_src + i * col);
            float4 data = src_float4[j];
            sum_val += exp(data.x - max_val) + exp(data.y - max_val) + exp(data.z - max_val) + exp(data.w - max_val);
        }
        if(gtid == 0) printf("\n%f\n", sum_val);
        for (int offset = block_step / 2/col_pre_thread; offset > 0; offset >>= 1) {
            sum_val += __shfl_down_sync(0xFFFFFFFF, sum_val, offset);
        }

        // Store the sum value only in the first thread of the block.
        sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);
        if(gtid == 0) printf("\n%f\n", sum_val);
        // Finally compute and store the softmax value.
        for (int j = tid; j < col/4; j += block_step) {
            float4* src_float4 = reinterpret_cast<float4*>(ptr_src + i * col);
            float4* dst_float4 = reinterpret_cast<float4*>(ptr_dst + i * col);
            float4 data = src_float4[j];
            dst_float4[j] = make_float4(exp(data.x - max_val) / sum_val, exp(data.y - max_val) / sum_val, exp(data.z - max_val) / sum_val, exp(data.w - max_val) / sum_val);
        }
    }
}




int main() {
    const int rows = 1000;
    const int cols = 8192;

    std::vector<float> input_data(rows * cols, 1);
    std::vector<float> output_data_cpu(rows * cols, 0);
    std::vector<float> output_data_gpu(rows * cols, 0);

    softmax_cpu<float>(input_data.data(), output_data_cpu.data(), rows, cols);

    dim3 grids(256, (rows/256 + 31) /32 *32); //(256, 32)

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