#include <cuda_runtime.h>
#include<iostream>
#include<random>
#include<algorithm>
#define SIZE 1024*256
__device__ __host__ unsigned int convert(float val){
    unsigned int cmp = *reinterpret_cast<unsigned int *>(&val);
    //cmp & (1<<31)比较最高位是否为1（负数），1<<32 = 0x80000000(最高位是1其余位是0)
    //负数按位取反~(cmp)
    unsigned int ret = (cmp & (1<<31)) ? ~(cmp) : (cmp | 0x80000000);
    return ret;
}

__device__ __host__ float deconvert(unsigned int val){
    //(val ^ 0x80000000)异或操作，正数的时候最高位convert为1，0x80000000最高位也是1，返回0
    unsigned int ret = (val & (1<<31)) ? (val ^ 0x80000000) : ~(val);
    return *reinterpret_cast<float*>(&ret);
}




__global__ void RadixSort_gpu(float* val, unsigned int* sort_tmp0, unsigned int* sort_tmp1, unsigned int N){
    int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 256;
    if(idx >= N) {
    return ;
  }
    for(int bit=0; bit<32; bit++){
        unsigned int mask = 1<<bit;
        unsigned count0 = 0, count1 = 0;
        for(int i=tid; i<N; i+=blockDim.x){
            unsigned int elem = (bit == 0) ? convert(val[i]) : sort_tmp0[i];
            if(elem & mask){
                sort_tmp1[count1+tid] = elem;
                count1+=blockDim.x;
            }else{
                sort_tmp0[count0+tid] = elem;
                count0+=blockDim.x;
            }
        }
        for(int i=0; i<count1; i+=blockDim.x){
            sort_tmp0[count0 + i+tid] = sort_tmp1[i+tid];
        }
    }
    //merge
    __shared__ unsigned int list_index[1024];//blockDim.x = 1024
    __shared__ unsigned int min_val, min_tid;
    unsigned int elem = 0xffffffff;
    list_index[tid] = 0;
    __syncthreads();

    for(unsigned int i=0; i<N; i++){
        
        unsigned int x = (list_index[tid] * blockDim.x + tid);
        // elem = sort_tmp0[x];
        if(x < N) {
            elem = sort_tmp0[x];
        }
        else {
            elem = 0xffffffff;
        }

        __syncthreads();
        min_val = min_tid = 0xffffffff;
        atomicMin(&min_val, elem);
        __syncthreads();
        if(min_val == elem){
            // min_tid = tid; //如果有多个最小值，取线程号最小的
            atomicMin(&min_tid, tid);
        }
        __syncthreads();
        if(min_tid == tid){
            list_index[tid]++;
            val[i] = deconvert(min_val);
        }

    }
}



void RadixSort_cpu(float* val, unsigned int size){
    unsigned int sort_tmp0[size],sort_tmp1[size];
    for(int bit=0; bit<32; bit++){
        unsigned int mask = 1<<bit;
        unsigned int count0 = 0, count1 = 0;
        for(int i=0; i<size; i++){
            unsigned int elem = (bit == 0) ? convert(val[i]) : sort_tmp0[i];
            if(elem & mask){
                sort_tmp1[count1++] = elem;
            }else{
                sort_tmp0[count0++] = elem;
            }
        }
        for(int i=0; i<count1; i++){
            sort_tmp0[count0 + i] = sort_tmp1[i];
        }
    }
    for(int i=0; i<size; i++){
        val[i] = deconvert(sort_tmp0[i]);
    }
}


int main(){
    float data[SIZE], data_gpu_host[SIZE];
    for(int i = 0;i < SIZE;i++) {
      data[i] = pow(-1,i) * (random()%1000);
    }

    float* data_cpu = data;
    RadixSort_cpu(data_cpu, SIZE);

    float* data_gpu_d;
    unsigned int *sort_tmp0, *sort_tmp1;
    cudaMalloc((void**)&data_gpu_d, SIZE * sizeof(float));
    cudaMalloc((void**)&sort_tmp0, SIZE * sizeof(unsigned int));
    cudaMalloc((void**)&sort_tmp1, SIZE * sizeof(unsigned int));
    cudaMemcpy(data_gpu_d, data, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    // dim3 blocks(1024);
    dim3 blocks(1024,1);
    dim3 grids( 1, 1);
    //warmup
    RadixSort_gpu<<<grids,blocks>>>(data_gpu_d, sort_tmp0, sort_tmp1, 1);
    cudaDeviceSynchronize();
    float millionseconds;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    RadixSort_gpu<<<grids,blocks>>>(data_gpu_d, sort_tmp0, sort_tmp1, SIZE);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&millionseconds,start,end);
    cudaDeviceSynchronize();
    printf("quick_sort_by_single_thread Elapsed Time: %f\n", millionseconds);

    
    // cudaMallocHost((void**)&data_gpu_host, SIZE*sizeof(float));
    cudaMemcpy(data_gpu_host, data_gpu_d, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < SIZE; i++) {
        if (fabs(data_cpu[i] - *(reinterpret_cast<float*>(data_gpu_host) + i)) > 1e-5) {
            success = false;
            printf(" i =%d, output_data_cpu = %f, output_data_gpu = %f\n", i,
                     data_cpu[i], 
                     *(reinterpret_cast<float*>(data_gpu_host) + i));
            break;
        }
    }

    if(success) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%f ",data_cpu[i]);
        }
        printf("\n");
        for(int i=0;i<10;i++){
            printf("%f ",*(reinterpret_cast<float*>(data_gpu_host) + i));
        }
        printf("\n");
    }
    return 0;
}