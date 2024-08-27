#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <algorithm> // For std::generate
#include <iostream>
#include <random>    // For std::default_random_engine and std::uniform_int_distribution

int main() {
    // 创建一个主机向量并初始化
    thrust::host_vector<int> h_vec(1000000);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // 将主机向量复制到设备向量
    thrust::device_vector<float> d_vec = h_vec;

    cudaEvent_t start, end;
    float millionseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    // 对设备向量进行排序
    thrust::sort(d_vec.begin(), d_vec.end());

    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&millionseconds, start, end);
    printf("Elapsed Time: %f\n", millionseconds);

    // 将排序后的设备向量复制回主机向量
    h_vec = d_vec;

    // 打印排序后的结果
    // for (const auto& val : h_vec) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
