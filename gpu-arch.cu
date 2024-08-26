#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "Major revision number: " << prop.major << "\n";
        std::cout << "Minor revision number: " << prop.minor << "\n";
    }
    return 0;
}