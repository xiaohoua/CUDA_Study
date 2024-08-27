#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define MAX_DEPTH 16

__device__ void swap(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int partition(float* arr, int low, int high) {
    float pivot = arr[high];
    int i = low;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[high]);
    return i;
}

__global__ void quickSortDevice(float* arr, int* stack, int length) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stackSize = 0;
    stack[stackSize++] = 0;
    stack[stackSize++] = length - 1;

    while (stackSize > 0) {
        int high = stack[--stackSize];
        int low = stack[--stackSize];

        if (low < high) {
            int pi = partition(arr, low, high);

            if (pi - 1 > low) {
                stack[stackSize++] = low;
                stack[stackSize++] = pi - 1;
            }

            if (pi + 1 < high) {
                stack[stackSize++] = pi + 1;
                stack[stackSize++] = high;
            }
        }
    }
}

void quick_sort_gpu(float* data, int length) {
    float* deviceArray;
    int* deviceStack;

    cudaMalloc(&deviceArray, length * sizeof(float));
    cudaMalloc(&deviceStack, 2 * MAX_DEPTH * sizeof(int));

    cudaMemcpy(deviceArray, data, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, end;
    float millionseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    quickSortDevice<<<1, 1>>>(deviceArray, deviceStack, length);

    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&millionseconds, start, end);
    printf("Elapsed Time: %f\n", millionseconds);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(data, deviceArray, length * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceArray);
    cudaFree(deviceStack);
}

int main() {
    int length = 1000000;
    float* hostArray = (float*)malloc(length * sizeof(float));

    // Initialize the array with random numbers
    for (int i = 0; i < length; ++i) {
        hostArray[i] = (float)(rand() % 10000);
    }

    // Call the sorting function
    quick_sort_gpu(hostArray, length);

    // Check the sorted array
    for (int i = 0; i < length - 1; ++i) {
        if (hostArray[i] > hostArray[i + 1]) {
            printf("Error in sorting\n");
            break;
        }
    }

    free(hostArray);

    return 0;
}
