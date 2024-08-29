#include<cuda_runtime.h>
#include<iostream>
#include <random>
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
__device__ void swap(T* a, T* b){
    T temp = *a;
    *a = *b;
    *b = temp;
}
template<typename T>
__device__ int partition_gpu(T* data, int low, int high){
    T pivot = data[low];
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
#define STACK_SIZE 100
template<typename T>
__global__ void quick_sort_gpu(T* src, int low, int high){
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
                __syncthreads();
                stack[++top] = high;
                __syncthreads();
            }
            high = pivot_index - 1;
        }
        
        if(top == -1) break;

        //先赋值再--
        high = stack[top--];
        __syncthreads();
        low = stack[top--];
        __syncthreads();
    }
    
}


int main() {
     const int size = 10000;
    int *output_data_cpu = new int[size];
    int *output_data_gpu = new int[size];
    int* device_data;

    // Initialize random data
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 10000);
    for (int i = 0; i < size; ++i) {
        output_data_cpu[i] = distribution(generator);
    }
    
    // Copy data to device
    // cudaMalloc((void**)&device_data, size * sizeof(int));
    cudaMalloc(&device_data, size*sizeof(int));
    cudaMemcpy(device_data, output_data_cpu, size * sizeof(int), cudaMemcpyHostToDevice);

    // Sort on CPU
    quick_sort_cpu(output_data_cpu, 0, size - 1);

    // Sort on GPU
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    //warmup
    quick_sort_gpu<<<1, 1>>>(device_data, 0, size - 1);
    float millionseconds;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    quick_sort_gpu<<<1, 1>>>(device_data, 0, size - 1);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&millionseconds,start,end);
    cudaDeviceSynchronize();
    printf("Elapsed Time: %f\n", millionseconds);

    // Copy sorted data back from device
    cudaMemcpy(output_data_gpu, device_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_data);
    // Check if CPU and GPU results are the same
    bool success = true;
    for (int i = 0; i < size; i++) {
        if (fabs((output_data_cpu[i] - output_data_gpu[i])) > 1e-5) {
            success = false;
            printf(" i =%d, output_data_cpu = %d, output_data_npu = %d\n", i, output_data_cpu[i], output_data_gpu[i]);
            break;
        }
    }

    if(success) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%d ",output_data_gpu[i]);
        }
        printf("\n");
        for(int i=0;i<10;i++){
            printf("%d ",output_data_cpu[i]);
        }
        printf("\n");
    }
    delete[] output_data_cpu;
    return 0;
}