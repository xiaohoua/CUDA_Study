#include <cstdio>
#include <iostream>

#include "cuda_runtime_api.h"


#define MAX_DEPTH 32
#define INSERTION_SORT 32

#define GPU_CHECK(ans)                                                         \
    { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
};

__device__ void selection_sort(unsigned int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            unsigned val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right,
                                   int depth) {
    // 当递归的深度大于设定的MAX_DEPTH或者待排序的数组长度小于设定的阈值，直接调用简单选择排序
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *left_ptr = data + left;
    unsigned int *right_ptr = data + right;
    unsigned int pivot = data[(left + right) / 2];
    // partition
    while (left_ptr <= right_ptr) {
        unsigned int left_val = *left_ptr;
        unsigned int right_val = *right_ptr;

        while (left_val < pivot) { // 找到第一个比pivot大的
            left_ptr++;
            left_val = *left_ptr;
        }

        while (right_val > pivot) { // 找到第一个比pivot小的
            right_ptr--;
            right_val = *right_ptr;
        }

        // do swap
        if (left_ptr <= right_ptr) {
            *left_ptr++ = right_val;
            *right_ptr-- = left_val;
        }
    }

    // recursive
    int n_right = right_ptr - data;
    int n_left = left_ptr - data;
    // Launch a new block to sort the the left part.
    if (left < (right_ptr - data)) {
        cudaStream_t l_stream;
        // 设置非阻塞流
        cudaStreamCreateWithFlags(&l_stream, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, l_stream>>>(data, left, n_right,
                                                    depth + 1);
        cudaStreamDestroy(l_stream);
    }

    // Launch a new block to sort the the right part.
    if ((left_ptr - data) < right) {
        cudaStream_t r_stream;
        // 设置非阻塞流
        cudaStreamCreateWithFlags(&r_stream, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, r_stream>>>(data, n_left, right,
                                                    depth + 1);
        cudaStreamDestroy(r_stream);
    }
}

// Call the quicksort kernel from the host.
void run_qsort(unsigned int *data, unsigned int nitems) {
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    GPU_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    float millionseconds;
    int left = 0;
    int right = nitems - 1;
    //warmup
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
    GPU_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&millionseconds, start, end);
    printf("Elapsed Time: %f\n", millionseconds);

    
}

// Initialize data on the host.
void initialize_data(unsigned int *dst, unsigned int nitems) {
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (unsigned i = 0; i < nitems; i++)
        dst[i] = rand() % nitems;
}

// Verify the results.
void print_results(int n, unsigned int *results_d) {
    unsigned int *results_h = new unsigned[n];
    GPU_CHECK(cudaMemcpy(results_h, results_d, n * sizeof(unsigned),cudaMemcpyDeviceToHost));
    std::cout << "Sort data : ";
    for (int i = 1; i < n; ++i){
        std::cout << results_h[i] << " ";
        // if (results_h[i - 1] > results_h[i]) {
        //     std::cout << "Invalid item[" << i - 1 << "]: " << results_h[i - 1]
        //               << " greater than " << results_h[i] << std::endl;
        //     exit(EXIT_FAILURE);
        // }
    }
    std::cout << std::endl;
    delete[] results_h;
}

void check_results(int n, unsigned int *results_d)
{
    unsigned int *results_h = new unsigned[n];
    GPU_CHECK(cudaMemcpy(results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 1 ; i < n ; ++i)
        if (results_h[i-1] > results_h[i])
        {
            std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "OK" << std::endl;
    delete[] results_h;
}

// Main entry point.

int main(int argc, char **argv) {
    int num_items = 1000000;
    // bool verbose = true;
    bool verbose = false;

    // Find/set device and get device properties
    int device = 1;
    cudaDeviceProp deviceProp;
    GPU_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    if (!(deviceProp.major > 3 ||
          (deviceProp.major == 3 && deviceProp.minor >= 5))) {
        printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.",
            device, deviceProp.name);
        return 0;
    }

    // Create input data
    unsigned int *h_data = 0;
    unsigned int *d_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data = (unsigned int *)malloc(num_items * sizeof(unsigned int));
    initialize_data(h_data, num_items);

    if (verbose) {
       std::cout << "Raw  data : ";
        for (int i = 0; i < num_items; i++)
            std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Allocate GPU memory.
    GPU_CHECK(cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
    GPU_CHECK(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int),cudaMemcpyHostToDevice));

    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    run_qsort(d_data, num_items);

    // print result
    if (verbose) print_results(num_items, d_data);
    // check result
    std::cout << "Validating results: ";
    check_results(num_items, d_data);
    free(h_data);
    GPU_CHECK(cudaFree(d_data));

    return 0;
}

/* 
编译：nvcc -o quick_sort_by_cuda --gpu-architecture=sm_70 -rdc=true quick_sort_by_cuda.cu
*/
