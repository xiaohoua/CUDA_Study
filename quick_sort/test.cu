#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define INSERTION_SORT 32
#define MAX_DEPTH 16

template<typename T>
int partition(T* data, int low, int high){
    int mid = low + (high - low) / 2;
    T pivot = data[mid];
    std::swap(data[mid], data[low]);
    while (low < high)
    {
        while(data[high] >= pivot && low<high){
            high--;
        }
        data[low] = data[high];
        while(data[low] <= pivot && low<high){
            low++;
        }
        data[high] = data[low];
    }
    
    data[low] = pivot;
    return low;
}
template<typename T>
void quick_sort_cpu(T* data, int low, int high){
    
    if(low < high){
        int i = partition(data, low ,high);
        quick_sort_cpu(data, low, i-1);
        quick_sort_cpu(data, i+1, high);
    }
}

template<typename T>
__device__ void swap(T* a, T* b) {
    T temp = *a;
    *a = *b;
    *b = temp;
}

template<typename T>
__device__ int partition_gpu(T* data, int low, int high) {
    T pivot = data[low];
    int i = low;
    int j = high;

    while (true) {
        while (data[i] < pivot) {
            i++;
        }
        while (data[j] > pivot) {
            j--;
        }
        if (i >= j) {
            return j;
        }
        swap(&data[i], &data[j]);
        i++;
        j--;
    }
}

template<typename T>
__device__ void selection_sort(T *data, int low, int high){
    for(int i=low; i<high; ++i){
        T min = data[i];
        int min_index = i;
        for(int j=i+1; j<=high; ++j){

            if(data[j] < min){
                min_index = j;
                min = data[j];
            }
        }
        if(low != min_index) {
            data[min_index] = data[i];
            data[i] = min;
        };
    }
}

template<typename T>
__global__ void quick_sort_dynamic_parallel(T* data, int low, int high, int depth) {
    if (low < high) {
        if(depth >= MAX_DEPTH || high - low <=INSERTION_SORT) {
            selection_sort(data, low, high);
            return;
        }
        int pi = partition_gpu(data, low, high);

        // Launch new kernels for the two partitions
        quick_sort_dynamic_parallel<<<1, 1>>>(data, low, pi, depth+1);
        quick_sort_dynamic_parallel<<<1, 1>>>(data, pi + 1, high, depth+1);

        // Synchronize to ensure the child kernels complete before the parent continues
        // cudaStreamSynchronize(stream);
    }
}
template<typename T>
__global__ void quick_sort_multi_stream(T* data, int low, int high, int depth) {
    if (low < high) {
        if(depth >= MAX_DEPTH || high - low <=INSERTION_SORT) {
            selection_sort(data, low, high);
            return;
        }
        int pi = partition_gpu(data, low, high);

        cudaStream_t left_stream, right_stream;
        cudaStreamCreateWithFlags(&left_stream, cudaStreamNonBlocking);
        // Launch new kernels for the two partitions
        quick_sort_multi_stream<<<1, 1, 0 ,left_stream>>>(data, low, pi, depth+1);
        cudaStreamDestroy(left_stream);
        cudaStreamCreateWithFlags(&right_stream, cudaStreamNonBlocking);
        quick_sort_multi_stream<<<1, 1, 0,right_stream>>>(data, pi + 1, high, depth+1);
        cudaStreamDestroy(right_stream);

    }
}

#define STACK_SIZE 100
template<typename T>
__global__ void quick_sort_single_thread(T* src, int low, int high){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int stack[STACK_SIZE];
    __shared__ int top;
    if(threadIdx.x == 0){
        top = -1;
    }

    while (true)
    {
        while (low < high)
        {
            int pivot_index = partition_gpu(src, low, high);
            if(pivot_index+1 < high){
                //先top++再存储
                stack[++top] = pivot_index + 1;
                // __syncthreads();
                stack[++top] = high;
                // __syncthreads();
            }
            high = pivot_index - 1;
        }
        
        if(top == -1) break;

        //先赋值再--
        high = stack[top--];
        // __syncthreads();
        low = stack[top--];
        // __syncthreads();
    }
    
}

void run_test_performance(int size){
    float start_time_v1 = 0,start_time_v2 = 0,start_time_v3 = 0;
    int *output_data_cpu = new int[size];
    int *output_data_gpu1 = new int[size];
    int *output_data_gpu2 = new int[size];
    int *output_data_gpu3 = new int[size];
    int* device_data1;
    int* device_data2;
    int* device_data3;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 10000);
    for (int i = 0; i < size; ++i) {
        output_data_cpu[i] = distribution(generator);
    }

    cudaMalloc(&device_data1, size*sizeof(int));
    cudaMalloc(&device_data2, size*sizeof(int));
    cudaMalloc(&device_data3, size*sizeof(int));
    cudaMemcpy(device_data1, output_data_cpu, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data2, output_data_cpu, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data3, output_data_cpu, size * sizeof(int), cudaMemcpyHostToDevice);

    quick_sort_cpu(output_data_cpu, 0, size);
    //warmup
    quick_sort_dynamic_parallel<<<1,1>>>(device_data1, 0, size, 0);
    quick_sort_multi_stream<<<1,1>>>(device_data2, 0, size, 0);
    quick_sort_single_thread<<<1,1>>>(device_data3, 0, size);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    quick_sort_dynamic_parallel<<<1,1>>>(device_data1, 0, size, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v1, start, stop);
    
    cudaEventRecord(start);
    quick_sort_multi_stream<<<1,1>>>(device_data2, 0, size, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v2, start, stop);
    
    // 版本3
    cudaEventRecord(start);
    quick_sort_single_thread<<<1,1>>>(device_data3, 0, size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&start_time_v3, start, stop);

    
    cudaMemcpy(output_data_gpu1,device_data1, size*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(output_data_gpu2,device_data2, size*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(output_data_gpu3,device_data3, size*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(device_data1);
    cudaFree(device_data2);
    cudaFree(device_data3);

    delete[] output_data_cpu;
    delete[] output_data_gpu1;
    delete[] output_data_gpu2;
    delete[] output_data_gpu3;

    std::cout << "quick_sort_dynamic_parallel Elapsed Time: " << start_time_v1 << " ms" << std::endl;
    std::cout << "quick_sort_multi_stream 2 Elapsed Time: " << start_time_v2 << " ms" << std::endl;
    std::cout << "quick_sort_single_thread Elapsed Time: " << start_time_v3 << " ms" << std::endl;
}


int main() {
    std::vector<int> size = {500, 2000, 5000, 10000, 100000, 1000000};
    std::vector<int> size2 = {10000, 100000, 1000000};
    for (int c : size2) {
        run_test_performance(c);
    }
    return 0;
}
